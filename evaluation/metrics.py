"""
Automatic evaluation of one generated literature review (no manual labels).

Evidence policy
---------------
Claims are checked ONLY against primary source text (abstract + extracted
full text) of the documents the review cites - never against the system's
own LLM-extracted notes. References are resolved to documents as follows:
  * title matches a benchmark paper  -> that paper's abstract + full text
  * title found in OpenAlex          -> the real paper's OpenAlex abstract
  * otherwise                        -> unverifiable (no evidence)
so a fabricated reference can never support a claim.

Metric families
---------------
A. Citation integrity  : reference validity, citation validity, source coverage
B. Grounding           : judge groundedness, hallucination (unsupported) rate,
                         contradiction rate, NLI entailment rate (local DeBERTa)
C. Cross-paper synthesis: multi-citation rate, grounded comparisons, pair coverage
D. Research gaps       : gap grounding, specificity, novelty, citation rate
E. Structure           : project's deterministic quality score, over-claims, length
"""

from __future__ import annotations

import hashlib
import re
import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from claim_checking import build_claim_items, judge_claims, _extract_json
from grounding import (
    CERTAINTY_PHRASES,
    COMPARATIVE_MARKERS,
    JUDGE_NLI_MODEL,
    EvidenceIndex,
    claim_support,
    get_embedder,
    get_nli,
    parse_reference_entries,
    parse_reference_title,
    review_sentences,
    split_body_and_references,
    title_similarity,
    word_count,
)
from llm_client import invoke_text, make_llm

from .config import JUDGE_MAX_CLAIMS, JUDGE_MODEL, JUDGE_SEED
from .openalex import find_doi, find_title

NLI_THRESHOLD = 0.5


# Takes a paper's primary text only (title, authors, year, abstract and full text, if it was really extracted)
# - no LLM notes - so claims are checked against the real source.
def _raw_doc(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Primary text only (no LLM analysis)."""
    return {
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "year": paper.get("year") or paper.get("published_year"),
        "abstract": paper.get("abstract", ""),
        "full_text": paper.get("full_text", "") if paper.get("content_source") == "full_text" else "",
    }


# Checks every reference of a generated review.
# For each [n] entry it extracts the title and decides:
#  - corpus: it matches (>= 0.85 similarity) a benchmark paper -> use that paper's text as evidence,
#  - openalex: found online by DOI or title -> use the real abstract,
#  - unverified/missing: probably invented -> it can never support a claim.
# Returns the evidence documents and the status of every reference.
def resolve_references(review: str, topic_papers: List[Dict[str, Any]], verify_online: bool = True):
    """Map each reference number to an evidence document and a verification status."""
    _, refs_text = split_body_and_references(review)
    entries = parse_reference_entries(refs_text)
    count = max(entries) if entries else 0
    docs, info = [], []
    for number in range(1, count + 1):
        entry = entries.get(number, "")
        title = parse_reference_title(entry) if entry else ""
        status, doc, corpus_index = "missing", {"title": ""}, None
        if title:
            scores = [title_similarity(title, p.get("title", "")) for p in topic_papers]
            best = int(np.argmax(scores)) if scores else -1
            if best >= 0 and scores[best] >= 0.85:
                status, doc, corpus_index = "corpus", _raw_doc(topic_papers[best]), best
            elif verify_online:
                doi = re.search(r"10\.\d{4,9}/[^\s,;]+", entry.replace("‑", "-"))
                found = find_doi(doi.group(0), title) if doi else None
                found = found or find_title(title)
                if found:
                    status, doc = "openalex", _raw_doc(found)
                else:
                    status = "unverified"
            else:
                status = "unverified"
        docs.append(doc)
        info.append({"number": number, "title": title, "status": status, "corpus_index": corpus_index})
    return docs, info


# Chooses which sentences the LLM judge will check (to save tokens): all gap sentences (up to 5)
# plus a random but repeatable sample of the others; the original order is kept.
def _sample_sentences(sentences, key: str, limit: int):
    """Deterministic stratified sample: all gap sentences (<=5) + random others."""
    rng = random.Random(int(hashlib.md5(f"{JUDGE_SEED}:{key}".encode()).hexdigest()[:8], 16))
    gap = [s for s in sentences if s.is_gap_section][:5]
    others = [s for s in sentences if not s.is_gap_section]
    rng.shuffle(others)
    chosen = gap + others[: max(0, limit - len(gap))]
    order = {id(s): i for i, s in enumerate(sentences)}
    return sorted(chosen, key=lambda s: order[id(s)])


GAP_PROMPT = """You assess research-gap statements from a literature review.
For each GAP, look at the EVIDENCE passages (excerpts from the source papers) and answer:
- specific: true if the gap names a concrete method, task, dataset, population, setting or metric
  (false for generic statements such as "more research is needed").
- addressed: true if the evidence shows a supplied paper already resolves this gap.

EVIDENCE:
{evidence}

GAPS:
{gaps}

Return ONLY JSON: {{"results": [{{"id": <gap id>, "specific": true|false, "addressed": true|false}}]}}"""


# Asks the judge LLM about the research-gap sentences: is each gap SPECIFIC and has it already been ADDRESSED
# in the evidence? Used for the gap specificity and novelty metrics.
def judge_gap_quality(gaps, index: EvidenceIndex, model: str = JUDGE_MODEL) -> Dict[int, Dict[str, bool]]:
    if not gaps:
        return {}
    pool, lines = [], []
    for gid, sentence in enumerate(gaps):
        passages = index.top_passages(sentence.claim, None, 2)
        refs = []
        for passage in passages:
            text = index.premise(passage)[:500]
            if text not in pool:
                pool.append(text)
            refs.append(f"E{pool.index(text) + 1}")
        lines.append(f"id={gid} | evidence: {', '.join(refs) or 'none'} | gap: {sentence.text}")
    prompt = GAP_PROMPT.format(
        evidence="\n".join(f"E{i + 1}: {t}" for i, t in enumerate(pool)) or "(none)",
        gaps="\n".join(lines),
    )
    llm = make_llm(model, temperature=0, max_tokens=1500, reasoning_effort="low")
    try:
        payload = _extract_json(invoke_text(llm, prompt, label="Gap Judge"))
        rows = payload.get("results", []) if isinstance(payload, dict) else payload
    except Exception:
        rows = []
    out = {}
    for row in rows:
        try:
            out[int(row["id"])] = {"specific": bool(row.get("specific")), "addressed": bool(row.get("addressed"))}
        except Exception:
            continue
    return out


# Safe division: numerator / denominator rounded to 4 decimals, or None when the denominator is 0.
def _rate(numerator: float, denominator: float) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


# Computes every automatic metric of one review, with no manual labels.
#  A. Citation integrity: reference validity, citation validity, citation density, source coverage.
#  C. Synthesis: how many sentences compare two or more papers.
#  B. Grounding: a local DeBERTa NLI model checks every sentence; an LLM judge labels a sample of claims
#     (SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED) -> groundedness and hallucination rate.
#  D. Research gaps: citation rate, grounding, specificity, novelty.
#  E. Structure: the rule-based score and over-claim counts.
# Finally it combines them into one overall quality index.
def evaluate_review(
    review: str,
    topic_papers: List[Dict[str, Any]],
    key: str,
    use_judge: bool = True,
    verify_online: bool = True,
) -> Dict[str, Any]:
    """Compute every automatic metric for one review. `key` seeds the claim sample."""
    from app import evaluate_generated_review  # project's deterministic structure evaluator

    docs, ref_info = resolve_references(review, topic_papers, verify_online)
    n_refs = len(docs)
    verified = {r["number"] for r in ref_info if r["status"] in {"corpus", "openalex"}}
    index = EvidenceIndex(docs, use_analysis=False)

    sentences = review_sentences(review)
    body_words = word_count(split_body_and_references(review)[0])
    cited = [s for s in sentences if s.citations]
    markers = [c for s in sentences for c in s.citations]
    valid_markers = [c for c in markers if c in verified]

    # ---------------- A. Citation integrity ----------------
    corpus_cited = {
        r["corpus_index"] for r in ref_info
        if r["status"] == "corpus" and any(r["number"] in s.citations for s in sentences)
    }
    metrics: Dict[str, Any] = {
        "word_count": body_words,
        "sentence_count": len(sentences),
        "reference_count": n_refs,
        "reference_validity": _rate(len(verified), n_refs) or 0.0,
        "citation_validity": _rate(len(valid_markers), len(markers)) or 0.0,
        "citation_density": _rate(len(cited), len(sentences)) or 0.0,
        "source_coverage": _rate(len(corpus_cited), len(topic_papers)) or 0.0,
    }

    # ---------------- C. Cross-paper synthesis ----------------
    multi = [s for s in sentences if len({c for c in s.citations if c in verified}) >= 2]
    comparisons = [s for s in multi if any(m in s.text.lower() for m in COMPARATIVE_MARKERS)]
    pairs = set()
    for s in multi:
        valid = sorted({c for c in s.citations if c in verified})
        pairs.update((a, b) for i, a in enumerate(valid) for b in valid[i + 1:])
    cited_valid = {c for c in valid_markers}
    possible_pairs = math.comb(len(cited_valid), 2) if len(cited_valid) >= 2 else 0
    metrics.update({
        "multi_citation_rate": _rate(len(multi), len(cited)) or 0.0,
        "grounded_comparisons": len(comparisons),
        "comparisons_per_1k_words": round(1000 * len(comparisons) / max(1, body_words), 3),
        "pair_coverage": _rate(len(pairs), possible_pairs) or 0.0,
    })

    # ---------------- B. Grounding (local NLI, all sentences) ----------------
    nli = get_nli(JUDGE_NLI_MODEL)
    nli_rows = []
    for s in sentences:
        valid = [c for c in s.citations if c in verified]
        if s.citations and not valid:
            entail, contra = 0.0, 0.0  # only fabricated/unresolvable citations
        else:
            support = claim_support(s.claim, index, nli, valid or None)
            entail, contra = support.entail, support.contradict
        nli_rows.append({"cited": bool(s.citations), "entail": round(entail, 4), "contradict": round(contra, 4)})
    cited_rows = [r for r in nli_rows if r["cited"]]
    metrics.update({
        "nli_entailment_rate": _rate(sum(r["entail"] >= NLI_THRESHOLD for r in cited_rows), len(cited_rows)) or 0.0,
        "nli_mean_entailment": round(float(np.mean([r["entail"] for r in cited_rows])), 4) if cited_rows else 0.0,
        "nli_unsupported_rate": _rate(sum(r["entail"] < NLI_THRESHOLD for r in nli_rows), len(nli_rows)) or 0.0,
        "nli_contradiction_rate": _rate(sum(r["contradict"] >= NLI_THRESHOLD for r in nli_rows), len(nli_rows)) or 0.0,
    })

    # ---------------- B. Grounding (LLM judge, sampled sentences) ----------------
    judged: List[Dict[str, Any]] = []
    if use_judge and sentences:
        sample = _sample_sentences(sentences, key, JUDGE_MAX_CLAIMS)
        items, pool = build_claim_items(sample, index)
        verdicts = judge_claims(items, pool, JUDGE_MODEL, label="Evaluation Judge")
        for item, sentence in zip(items, sample):
            verdict = verdicts.get(item.id, {})
            judged.append({
                "sentence": item.sentence,
                "cited": bool(item.cited),
                "gap": sentence.is_gap_section,
                "label": verdict.get("label", "UNJUDGED"),
                "reason": verdict.get("reason", ""),
                "nli_entail": nli_rows[sentences.index(sentence)]["entail"],
            })
    claims = [j for j in judged if j["label"] not in {"NO_CLAIM", "UNJUDGED"}]
    cited_claims = [j for j in claims if j["cited"]]

    # Counts how many judged rows have one of the given labels.
    def count(rows, *labels):
        return sum(1 for r in rows if r["label"] in labels)

    metrics.update({
        "judged_claims": len(claims),
        "groundedness": _rate(count(cited_claims, "SUPPORTED"), len(cited_claims)) or 0.0,
        "groundedness_lenient": _rate(
            count(cited_claims, "SUPPORTED") + 0.5 * count(cited_claims, "PARTIAL"), len(cited_claims)
        ) or 0.0,
        "hallucination_rate": _rate(count(claims, "UNSUPPORTED", "CONTRADICTED"), len(claims)) if claims else None,
        "contradiction_rate": _rate(count(claims, "CONTRADICTED"), len(claims)) if claims else None,
        "partial_rate": _rate(count(claims, "PARTIAL"), len(claims)) if claims else None,
    })
    if claims:
        agree = [
            (j["label"] == "SUPPORTED") == (j["nli_entail"] >= NLI_THRESHOLD)
            for j in claims
        ]
        metrics["judge_nli_agreement"] = round(float(np.mean(agree)), 4)

    # ---------------- D. Research gaps ----------------
    gap_sentences = [s for s in sentences if s.is_gap_section]
    gap_claims = [j for j in claims if j["gap"]]
    metrics.update({
        "gap_count": len(gap_sentences),
        "gap_citation_rate": _rate(sum(bool(s.citations) for s in gap_sentences), len(gap_sentences)) or 0.0,
        "gap_grounding": _rate(
            count(gap_claims, "SUPPORTED") + 0.5 * count(gap_claims, "PARTIAL"), len(gap_claims)
        ) or 0.0,
    })
    gap_quality = judge_gap_quality(gap_sentences[:6], index) if (use_judge and gap_sentences) else {}
    if gap_quality:
        metrics["gap_specificity"] = round(float(np.mean([v["specific"] for v in gap_quality.values()])), 4)
        metrics["gap_novelty"] = round(float(np.mean([not v["addressed"] for v in gap_quality.values()])), 4)
    else:
        metrics["gap_specificity"] = 0.0
        metrics["gap_novelty"] = 0.0

    # ---------------- E. Structure ----------------
    structural = evaluate_generated_review(review, {"raw_papers_list": topic_papers, "analyzed_papers": topic_papers})
    metrics.update({
        "structure_score": float(structural.get("overall_quality_score", 0.0)),
        "overclaim_count": sum(1 for s in sentences for p in CERTAINTY_PHRASES if p in s.text.lower()),
        "redundancy_ratio": structural.get("redundancy", {}).get("ratio", 0.0),
    })

    # ---------------- Composite ----------------
    components = [
        metrics["reference_validity"],
        metrics["citation_validity"],
        metrics["groundedness"],
        1.0 - (metrics["hallucination_rate"] if metrics["hallucination_rate"] is not None else 1.0),
        metrics["source_coverage"],
        min(1.0, metrics["multi_citation_rate"] / 0.3),
        metrics["gap_grounding"],
        metrics["structure_score"] / 100.0,
    ]
    metrics["overall_quality_index"] = round(float(np.mean(components)), 4)

    details = {"references": ref_info, "judged_claims": judged, "nli": nli_rows, "gap_quality": gap_quality}
    return {"metrics": metrics, "details": details}


# ----------------------------------------------------------------------
# Pipeline-level gap verification (Skeptic) evaluation
# ----------------------------------------------------------------------

# Evaluates the Skeptic/Verifier: judges every PROPOSED gap and every VERIFIED gap against the raw text
# of its supporting papers. It reports precision of all proposals (what M3 accepts), precision of verified gaps
# (what M4 accepts), the rejection rate, and 'skeptic precision' = share of rejected gaps that really were
# unsupported.
def evaluate_gap_candidates(
    gap_analysis: Dict[str, Any],
    analyzed_papers: List[Dict[str, Any]],
    topic_papers: List[Dict[str, Any]],
    use_judge: bool = True,
) -> Dict[str, Any]:
    """
    Judge every proposed and every verified gap statement against the raw text of
    its supporting papers. Measures what the Skeptic/Verifier contributes:
      precision_unfiltered = supported share of all proposer gaps (what M3 accepts)
      precision_verified   = supported share of verifier-approved gaps (what M4 accepts)
      skeptic_precision    = share of REJECTED gaps that were indeed not supported
    """
    from grounding import ReviewSentence

    titles = [str(p.get("title", "")).strip() for p in analyzed_papers]
    index = EvidenceIndex([_raw_doc(p) for p in topic_papers], use_analysis=False)

    # Turns a gap into a review-style sentence with citation numbers, found by matching its supporting
    # paper titles to the paper list, so the normal claim checker can judge it.
    def as_sentence(gap):
        cited = []
        for title in gap.get("supporting_papers", []):
            scores = [title_similarity(title, t) for t in titles]
            if scores and max(scores) >= 0.85:
                cited.append(int(np.argmax(scores)) + 1)
        text = str(gap.get("gap_statement", "")).strip()
        marker = "".join(f" [{c}]" for c in cited)
        return ReviewSentence(text=text + marker, section="Research Gaps", citations=cited, is_gap_section=True, is_conclusion=False)

    proposed = gap_analysis.get("proposed_gaps", []) or []
    verified = gap_analysis.get("verified_gaps", []) or []
    if not proposed or not use_judge:
        return {"proposed": len(proposed), "verified": len(verified)}

    sentences = [as_sentence(g) for g in proposed] + [as_sentence(g) for g in verified]
    items, pool = build_claim_items(sentences, index)
    verdicts = judge_claims(items, pool, JUDGE_MODEL, label="Gap Candidate Judge")
    labels = [verdicts.get(i.id, {}).get("label", "UNJUDGED") for i in items]
    prop_labels, ver_labels = labels[: len(proposed)], labels[len(proposed):]

    # Share of judged gaps that are supported: SUPPORTED counts 1, PARTIAL counts 0.5; unjudged ones are ignored.
    def precision(rows):
        rows = [r for r in rows if r not in {"NO_CLAIM", "UNJUDGED"}]
        return _rate(sum(r == "SUPPORTED" for r in rows) + 0.5 * sum(r == "PARTIAL" for r in rows), len(rows))

    # A proposed gap counts as rejected when no verified gap is semantically close to it.
    embedder = get_embedder()
    prop_vec = embedder.encode([g.get("gap_statement", "") for g in proposed], normalize_embeddings=True, show_progress_bar=False)
    ver_vec = (
        embedder.encode([g.get("gap_statement", "") for g in verified], normalize_embeddings=True, show_progress_bar=False)
        if verified else np.zeros((0, prop_vec.shape[1]))
    )
    kept = (prop_vec @ ver_vec.T).max(axis=1) >= 0.75 if len(verified) else np.zeros(len(proposed), bool)
    rejected = [label for label, k in zip(prop_labels, kept) if not k]
    rejected_judged = [r for r in rejected if r not in {"NO_CLAIM", "UNJUDGED"}]

    return {
        "proposed": len(proposed),
        "verified": len(verified),
        "rejection_rate": _rate(len(proposed) - int(kept.sum()), len(proposed)),
        "precision_unfiltered": precision(prop_labels),
        "precision_verified": precision(ver_labels) if verified else None,
        "skeptic_precision": _rate(sum(r != "SUPPORTED" for r in rejected_judged), len(rejected_judged)),
        "labels_proposed": prop_labels,
        "labels_verified": ver_labels,
    }
