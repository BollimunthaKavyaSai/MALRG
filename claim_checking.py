"""
LLM claim checking against retrieved source evidence.

Used twice, with different models, so the system is never graded by the
same model that critiques it:
  * review_feedback.py   -> Fact-Checker agent inside the system (gpt-oss-20b)
  * evaluation/metrics.py -> independent judge for the ablation (Qwen)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from grounding import EvidenceIndex, ReviewSentence
from llm_client import invoke_text, make_llm

LABELS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED", "NO_CLAIM")


# One sentence of the review that is going to be fact-checked.
# Holds an id, the original sentence, the claim text (citation marks removed), the cited paper numbers,
# the ids of the evidence passages chosen for it, and whether it belongs to the research-gaps section.
@dataclass
class ClaimItem:
    id: int
    sentence: str
    claim: str
    cited: List[int]
    evidence_ids: List[int]
    is_gap: bool = False


# Prepares sentences for fact-checking by attaching the best evidence passages to each one.
#  - A sentence WITH citations gets passages only from the papers it cites (checks attribution).
#  - A sentence WITHOUT citations gets passages from any paper (checks that it is grounded at all).
# Returns the list of ClaimItems and an evidence pool (paper number + passage text), where the same
# passage is stored only once.
def build_claim_items(
    sentences: Sequence[ReviewSentence],
    index: EvidenceIndex,
    per_claim: int = 3,
):
    """
    Attach evidence passages to each sentence.
    Cited sentences get passages from their cited papers only (attribution);
    uncited sentences get passages from any source paper (corpus grounding).
    Returns (items, evidence_pool) where evidence_pool[i] = (paper_no, premise text).
    """
    pool: List[tuple] = []
    pool_key: Dict[str, int] = {}
    items: List[ClaimItem] = []

    for number, sentence in enumerate(sentences):
        valid_cited = [c for c in sentence.citations if 1 <= c <= index.n_papers]
        passages = []
        if valid_cited:
            per_paper = max(1, per_claim // len(valid_cited)) if len(valid_cited) <= per_claim else 1
            for paper in valid_cited[:4]:
                passages.extend(index.top_passages(sentence.claim, [paper], per_paper + (1 if len(valid_cited) == 1 else 0)))
        elif not sentence.citations:
            passages = index.top_passages(sentence.claim, None, per_claim)

        evidence_ids = []
        for passage in passages:
            text = index.premise(passage)
            if text not in pool_key:
                pool_key[text] = len(pool)
                pool.append((passage.paper, text))
            evidence_ids.append(pool_key[text])

        items.append(
            ClaimItem(
                id=number,
                sentence=sentence.text,
                claim=sentence.claim,
                cited=list(sentence.citations),
                evidence_ids=evidence_ids,
                is_gap=sentence.is_gap_section,
            )
        )
    return items, pool


# Finds the JSON in the judge model's answer: removes <think> tags and code fences,
# locates the first { or [ and parses from there.
def _extract_json(text: str):
    text = re.sub(r"(?s)<think>.*?</think>", "", text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text[start:])
    return payload


PROMPT = """You are a strict scientific fact-checker for literature reviews.

For every CLAIM below, decide whether the EVIDENCE passages it points to support it.
The evidence passages are verbatim excerpts or extracted notes from the source papers
(each prefixed with the authors and year of the paper it comes from).

Labels:
- SUPPORTED: every factual detail of the claim is stated in, or directly implied by, the evidence.
- PARTIAL: the core of the claim is supported but a detail (number, dataset, scope,
  attribution, strength of wording) is missing from or overstated relative to the evidence.
- UNSUPPORTED: the evidence does not back the claim, or the claim has no evidence.
- CONTRADICTED: the evidence states something incompatible with the claim.
- NO_CLAIM: the sentence makes no checkable factual statement (pure framing/transition).

A research-gap claim ("X has not been studied", "evaluation is limited to Y") is SUPPORTED
only if the evidence reports that limitation or clearly shows the missing aspect.
Judge ONLY against the shown evidence. Do not use outside knowledge.

EVIDENCE:
{evidence}

CLAIMS:
{claims}

Return ONLY JSON: {{"results": [{{"id": <claim id>, "label": "<LABEL>", "reason": "<max 15 words>"}}]}}
"""


# Asks an LLM 'judge' to label each claim against its evidence.
#  1. Works in batches of 6 claims to keep prompts small.
#  2. The prompt lists the evidence (E1, E2 ...) and the claims, and explains the labels:
#     SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED, NO_CLAIM.
#  3. Parses the JSON answer (one retry if it is not valid JSON).
#  4. Unknown labels become UNSUPPORTED; missing answers become UNJUDGED.
# Used twice with different models: as the Fact-Checker inside the system, and as the independent judge
# in the evaluation, so the system is never graded by the same model that checks it.
def judge_claims(
    items: Sequence[ClaimItem],
    pool: Sequence[tuple],
    model: str,
    batch_size: int = 6,
    label: str = "Claim Judge",
    max_evidence_chars: int = 650,
) -> Dict[int, Dict[str, str]]:
    """Return {item_id: {"label": ..., "reason": ...}} for every item."""
    llm = make_llm(model, temperature=0, max_tokens=3000, reasoning_effort="low")
    results: Dict[int, Dict[str, str]] = {}

    for start in range(0, len(items), batch_size):
        batch = list(items[start: start + batch_size])
        used = sorted({e for item in batch for e in item.evidence_ids})
        local_ids = {global_id: n + 1 for n, global_id in enumerate(used)}
        evidence_text = "\n".join(
            f"E{local_ids[g]} [paper {pool[g][0]}]: {pool[g][1][:max_evidence_chars]}" for g in used
        ) or "(no evidence available)"
        claims_text = "\n".join(
            f"id={item.id} | evidence: {', '.join('E' + str(local_ids[e]) for e in item.evidence_ids) or 'none'}"
            f" | claim: {item.sentence}"
            for item in batch
        )
        prompt = PROMPT.format(evidence=evidence_text, claims=claims_text)

        parsed: Optional[dict] = None
        for _attempt in range(2):
            raw = invoke_text(llm, prompt, label=label)
            try:
                parsed = _extract_json(raw)
                break
            except Exception:
                prompt += "\nYour previous answer was not valid JSON. Return only the JSON object."
        rows = (parsed or {}).get("results", []) if isinstance(parsed, dict) else (parsed or [])
        for row in rows:
            try:
                item_id = int(row.get("id"))
            except Exception:
                continue
            verdict = str(row.get("label", "")).upper().strip()
            if verdict not in LABELS:
                verdict = "UNSUPPORTED"
            results[item_id] = {"label": verdict, "reason": str(row.get("reason", ""))[:200]}
        for item in batch:
            results.setdefault(item.id, {"label": "UNJUDGED", "reason": "judge output missing"})
    return results
