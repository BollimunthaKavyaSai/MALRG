"""
LLM-as-judge evaluation (G-Eval style rubric), run with a model family that is
different from the writer (qwen vs. gpt-oss) to limit self-preference bias.
The judge is blind: it never sees which configuration produced a review.

  * rubric_judge      - 6 quality criteria (1-10) + per-gap labels (grounded/specific/novel)
  * claim_judge       - independent second opinion on sampled citation claims
  * relevance_judge   - graded relevance (0/1/2) of retrieved candidates -> silver qrels
  * candidate_judge   - validity of every proposed gap -> silver labels for the Skeptic
"""

import random
import re

import numpy as np

import config
from malrg import llm
from malrg.text_utils import body_sentences, parse_citations, strip_citations

RUBRIC = {
    "coverage": "Covers the important aspects of the topic using the provided sources.",
    "synthesis": "Integrates and compares sources (relations, agreements, contradictions) instead of listing papers one by one.",
    "critical_analysis": "Evaluates strengths, weaknesses and trade-offs of methods and evidence.",
    "gap_quality": "Research gaps are specific, actionable and justified by the sources' limitations.",
    "coherence": "Logical organization, clear thematic structure, readable academic prose.",
    "faithfulness": "Claims appear consistent with the provided sources; no invented facts or numbers.",
}


def _clip(text, n):
    return " ".join((text or "").split()[:n])


def source_digest(bench):
    out = []
    for p in bench["papers"]:
        s = p.get("sections", {})
        lim = s.get("limitations") or s.get("conclusion") or s.get("discussion") or ""
        out.append(f"[{p['ref_id']}] {p['title']} ({p.get('year')})\n"
                   f"  abstract: {_clip(p.get('abstract'), 85)}\n"
                   f"  limitations/conclusion: {_clip(lim, 55)}")
    return "\n".join(out)


def _review_for_judge(md, max_words=1500):
    body = re.split(r"\n#+\s*References", md, flags=re.I)[0]
    return _clip(body, max_words)


def rubric_judge(run, bench):
    crit = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    prompt = f"""You are an expert peer reviewer evaluating an automatically generated literature review.
Topic: "{bench['topic']}"

REFERENCE SOURCES (the papers the review is expected to draw on):
{source_digest(bench)}

REVIEW TO EVALUATE:
<<<
{_review_for_judge(run['review_md'])}
>>>

Score each criterion from 1 (very poor) to 10 (excellent). Be strict and calibrated.
{crit}

Then list every research gap stated in the review's gap section and label it:
grounded = the gap follows from limitations visible in the reference sources,
specific = concrete and actionable (not "more research is needed"),
novel = not already solved by one of the reference sources.

Return JSON:
{{"scores": {{"coverage": n, "synthesis": n, "critical_analysis": n, "gap_quality": n,
"coherence": n, "faithfulness": n}}, "justification": "2 sentences",
"gaps": [{{"gap": "short paraphrase", "grounded": true/false, "specific": true/false, "novel": true/false}}]}}"""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="judge", max_tokens=1200, default={})
    scores = (out or {}).get("scores", {}) if isinstance(out, dict) else {}
    clean = {}
    for k in RUBRIC:
        try:
            clean[f"judge_{k}"] = float(scores.get(k))
        except (TypeError, ValueError):
            clean[f"judge_{k}"] = np.nan
    vals = [v for v in clean.values() if not np.isnan(v)]
    clean["judge_overall"] = float(np.mean(vals)) if vals else np.nan
    gaps = [g for g in (out or {}).get("gaps", []) if isinstance(g, dict)] if isinstance(out, dict) else []
    good = [g for g in gaps if g.get("grounded") is True and g.get("specific") is True
            and g.get("novel") is True]
    clean["judge_gap_count"] = len(gaps)
    clean["judge_gap_precision"] = len(good) / len(gaps) if gaps else 0.0
    clean["judge_gap_grounded_rate"] = (sum(1 for g in gaps if g.get("grounded") is True) / len(gaps)
                                        if gaps else 0.0)
    return clean, out


def claim_judge(run, bench, rows, n_samples=6):
    """Independent LLM check of sampled cited claims against their best passage."""
    if not config.CONFIGS.get(run["config_id"], {}).get("retrieval", True):
        return {}
    from evaluation.nli import PassageIndex
    idx = PassageIndex(bench["papers"])
    cited = [r for r in rows if r["citations"] and all(1 <= c <= len(bench["papers"]) for c in r["citations"])]
    if not cited:
        return {"judge_claim_support": 0.0}
    rnd = random.Random(f"{bench['slug']}-{run['config_id']}")
    sample = rnd.sample(cited, min(n_samples, len(cited)))
    items = []
    for k, r in enumerate(sample):
        tops = idx.top_passages(strip_citations(r["sentence"]), r["citations"], k=1)
        ev = " ... ".join(_clip(t, 90) for _, t, _ in tops[:2])
        items.append(f"C{k}: CLAIM: {strip_citations(r['sentence'])}\n    EVIDENCE: {ev}")
    prompt = f"""For each claim decide if the evidence supports it.
"supported" = evidence states it; "partial" = only part of it; "unsupported" = not stated or contradicted.

{chr(10).join(items)}

Return JSON {{"labels": [{{"id": "C0", "label": "supported|partial|unsupported"}}]}}"""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="judge", max_tokens=500, default={})
    labels = [str(x.get("label", "")).lower() for x in (out or {}).get("labels", []) if isinstance(x, dict)]
    if not labels:
        return {}
    score = sum(1.0 if l == "supported" else 0.5 if l == "partial" else 0.0 for l in labels)
    nli_agree = sum(1 for l, r in zip(labels, sample)
                    if (l == "supported") == bool(r.get("supported")))
    return {"judge_claim_support": score / len(labels), "nli_judge_agreement": nli_agree / len(labels)}


def relevance_judge(bench, pool):
    lines = "\n".join(f"D{i}: {c['title']} -- {_clip(c.get('abstract'), 45)}" for i, c in enumerate(pool))
    prompt = f"""Grade how relevant each document is for writing a literature review on:
"{bench['topic']}"
2 = directly on topic, 1 = partially / adjacent, 0 = not relevant.

{lines}

Return JSON {{"grades": [{{"id": "D0", "grade": 0|1|2}}]}} covering every document."""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="judge", max_tokens=900, default={})
    grades = {}
    for g in (out or {}).get("grades", []):
        try:
            grades[int(str(g["id"])[1:])] = int(g["grade"])
        except (KeyError, ValueError, TypeError):
            continue
    return grades


def candidate_judge(bench, candidates):
    if not candidates:
        return {}
    lines = "\n".join(f"G{i}: {g.get('gap')} (cites {g.get('supporting_papers')})"
                      for i, g in enumerate(candidates))
    prompt = f"""Topic: "{bench['topic']}"
SOURCES:
{source_digest(bench)}

Label each candidate research gap as valid (true) only if it is supported by the sources' limitations,
specific, and not already solved by one of the sources.
{lines}

Return JSON {{"labels": [{{"id": "G0", "valid": true/false}}]}}"""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="judge", max_tokens=600, default={})
    labels = {}
    for g in (out or {}).get("labels", []):
        try:
            labels[int(str(g["id"])[1:])] = bool(g["valid"])
        except (KeyError, ValueError, TypeError):
            continue
    return labels
