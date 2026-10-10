"""
Stages 5-10 of the pipeline (Review-1 slides 23-25) plus the feedback agent.

  5. Paper Analysis Agent        - structured evidence + verbatim quotes, Evidence Gate
  6. Cross-Paper Synthesis Agent - themes, agreements, contradictions, trade-offs
  7. Adversarial Gap Triad       - Proposer -> Evidence Gate -> Skeptic -> Grounded Verifier
  8. Taxonomy Agent              - embedding clustering of papers into research themes
  9. Citation Structuring Agent  - deterministic IEEE references from verified metadata
 10. Citation-Grounded Writer    - one writer prompt shared by every ablation config
 11. Feedback Agent (novel)      - Verification-Guided Iterative Refinement (VGIR)
"""

import json
import re

import numpy as np

import config
from malrg import llm
from malrg.retrieval import embedder
from malrg.text_utils import (body_sentences, norm_for_match, parse_citations, quote_in_text,
                              split_markdown_sections, strip_citations)


# ---------------------------------------------------------------------------
# 5. Paper analysis + Evidence Gate
# ---------------------------------------------------------------------------
def _clip_words(text, n):
    w = (text or "").split()
    return " ".join(w[:n])


def paper_context(paper, budget=1250):
    s = paper.get("sections", {})
    parts = [("ABSTRACT", paper.get("abstract", ""), 180)]
    if s.get("method"):
        parts.append(("METHOD", s["method"], 300))
    if s.get("results"):
        parts.append(("RESULTS", s["results"], 380))
    for name in ("limitations", "discussion", "conclusion"):
        if s.get(name):
            parts.append((name.upper(), s[name], 220))
            break
    used = sum(min(n, len(t.split())) for _, t, n in parts)
    if used < 700:  # weak section detection: fall back to evenly spaced passages
        ps = paper["passages"]
        picks = [ps[i] for i in np.linspace(0, len(ps) - 1, num=min(8, len(ps)), dtype=int)]
        parts.append(("BODY EXCERPTS", " ... ".join(picks), budget - used))
    return "\n\n".join(f"[{h}]\n{_clip_words(t, n)}" for h, t, n in parts)


def analyze_paper(paper):
    prompt = f"""You are the Paper Analysis Agent. Extract structured evidence from ONE research paper.
Use ONLY the text below. If something is not reported, write "not reported".

TITLE: {paper['title']} ({paper.get('year')})
{paper_context(paper)}

Return JSON:
{{
 "problem": "research problem addressed (1 sentence)",
 "dataset": "datasets / data sources used",
 "method": "core technique or architecture (1-2 sentences)",
 "key_results": ["main findings, keep exact numbers"],
 "metrics": ["evaluation metrics used"],
 "limitations": ["limitations stated or evident"],
 "contributions": ["main contributions"],
 "evidence": [{{"field": "method|results|limitations|dataset", "quote": "an EXACT sentence copied verbatim from the text"}}]
}}
Give 4-6 evidence quotes copied character-for-character from the text."""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="analysis", max_tokens=1400, default={})
    if not isinstance(out, dict):
        out = {}
    # Evidence Gate: keep only quotes that really occur in the paper.
    text_norm = norm_for_match(" ".join(paper["passages"]) + " " + paper.get("abstract", ""))
    verified, rejected = [], []
    for ev in out.get("evidence", []) or []:
        if isinstance(ev, dict) and quote_in_text(ev.get("quote", ""), text_norm):
            verified.append(ev)
        else:
            rejected.append(ev)
    out["evidence"] = verified
    out["evidence_rejected"] = len(rejected)
    out["evidence_fidelity"] = round(len(verified) / max(1, len(verified) + len(rejected)), 3)
    out["ref_id"] = paper["ref_id"]
    out["title"] = paper["title"]
    return out


def compact_analysis(a, with_quotes=True):
    def lst(x, n=3):
        return "; ".join(str(i) for i in (x or [])[:n]) if isinstance(x, list) else str(x or "")
    lines = [
        f"[{a['ref_id']}] {a['title']}",
        f"  problem: {a.get('problem', '')}",
        f"  data: {a.get('dataset', '')}",
        f"  method: {a.get('method', '')}",
        f"  results: {lst(a.get('key_results'), 3)}",
        f"  metrics: {lst(a.get('metrics'), 4)}",
        f"  limitations: {lst(a.get('limitations'), 3)}",
    ]
    if with_quotes and a.get("evidence"):
        q = " | ".join(f"\"{_clip_words(e['quote'], 40)}\"" for e in a["evidence"][:2])
        lines.append(f"  verified quotes: {q}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Cross-paper synthesis
# ---------------------------------------------------------------------------
def cross_paper_analysis(topic, analyses):
    ctx = "\n".join(compact_analysis(a, with_quotes=False) for a in analyses)
    prompt = f"""You are the Cross-Paper Synthesis Agent for a literature review on "{topic}".
Compare the papers below with each other (do not summarize them one by one).

{ctx}

Return JSON:
{{
 "themes": [{{"name": "short theme name", "papers": [ids], "summary": "what unites these papers"}}],
 "agreements": [{{"finding": "...", "papers": [ids]}}],
 "contradictions": [{{"issue": "...", "papers": [ids], "explanation": "why results/conclusions differ"}}],
 "method_tradeoffs": [{{"tradeoff": "e.g. accuracy vs. cost", "papers": [ids]}}],
 "repeated_limitations": [{{"limitation": "...", "papers": [ids]}}],
 "unexplored": ["directions none of the papers address"]
}}
Use 2-4 themes. Every item must reference paper ids from the list."""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="cross_paper", max_tokens=1600, default={})
    return out if isinstance(out, dict) else {}


def compact_cross(cross):
    if not cross:
        return ""
    lines = []
    for key, field in (("themes", "name"), ("agreements", "finding"), ("contradictions", "issue"),
                       ("method_tradeoffs", "tradeoff"), ("repeated_limitations", "limitation")):
        for item in (cross.get(key) or [])[:4]:
            if isinstance(item, dict):
                extra = f" - {item.get('explanation')}" if item.get("explanation") else ""
                lines.append(f"- {key[:-1] if key.endswith('s') else key}: {item.get(field, '')}{extra} "
                             f"{item.get('papers', [])}")
    for u in (cross.get("unexplored") or [])[:3]:
        lines.append(f"- unexplored: {u}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Adversarial research-gap triad
# ---------------------------------------------------------------------------
def propose_gaps(topic, analyses, cross):
    lim = "\n".join(_gap_evidence(a) for a in analyses)
    cross_txt = compact_cross(cross) if cross else "(no cross-paper analysis available)"
    prompt = f"""You are the Gap Proposer. Topic: "{topic}".
Propose candidate research gaps from the paper limitations and cross-paper findings.

PAPER EVIDENCE:
{lim}

CROSS-PAPER FINDINGS:
{cross_txt}

Return JSON {{"gaps": [{{"gap": "specific, actionable research gap", "supporting_papers": [ids],
"evidence": "the limitation/finding this gap is derived from"}}]}} with 6 candidates."""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="gap_proposer", max_tokens=1300, default={})
    gaps = out.get("gaps", []) if isinstance(out, dict) else []
    return [g for g in gaps if isinstance(g, dict) and g.get("gap")]


def evidence_gate(gaps, analyses, papers):
    """Deterministic gate: each gap must cite valid papers and match their limitation evidence."""
    ids = {a["ref_id"] for a in analyses}
    lim_text = {a["ref_id"]: [str(x) for x in (a.get("limitations") or [])] +
                [str(x) for x in (a.get("key_results") or [])] for a in analyses}
    emb = embedder()
    passed, failed = [], []
    for g in gaps:
        sup = [int(x) for x in g.get("supporting_papers", []) if str(x).isdigit() and int(x) in ids]
        if not sup:
            failed.append({**g, "gate_reason": "no valid supporting paper"})
            continue
        cand_text = [t for r in sup for t in lim_text.get(r, [])]
        if not cand_text:
            failed.append({**g, "gate_reason": "supporting papers have no limitation evidence"})
            continue
        v = emb.encode([g["gap"] + " " + str(g.get("evidence", ""))] + cand_text, normalize_embeddings=True)
        sim = float((v[1:] @ v[0]).max())
        if sim < 0.35:
            failed.append({**g, "gate_reason": f"weak evidence link (sim={sim:.2f})"})
            continue
        passed.append({**g, "supporting_papers": sup, "gate_similarity": round(sim, 3)})
    return passed, failed


def skeptic_review(topic, gaps, analyses):
    ev = "\n".join(_gap_evidence(a) for a in analyses)
    gl = "\n".join(f"G{i}: {g['gap']} (papers {g.get('supporting_papers')}; basis: {g.get('evidence', '')})"
                   for i, g in enumerate(gaps))
    prompt = f"""You are the adversarial Skeptic in a research-gap verification pipeline (topic: "{topic}").
Challenge every candidate gap. Reject a gap if it is (a) already solved by one of the papers,
(b) not supported by the cited evidence, (c) vague or generic ("more research is needed"),
or (d) overstated. Otherwise accept it, or revise it to be specific and evidence-bound.

PAPERS:
{ev}

CANDIDATE GAPS:
{gl}

Return JSON {{"reviews": [{{"id": "G0", "verdict": "accept|revise|reject", "issue": "short critique",
"revised_gap": "only if verdict is revise"}}]}}"""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="gap_skeptic", max_tokens=1200, default={})
    reviews = out.get("reviews", []) if isinstance(out, dict) else []
    by_id = {}
    for r in reviews:
        if isinstance(r, dict) and str(r.get("id", "")).upper().startswith("G"):
            try:
                by_id[int(str(r["id"])[1:])] = r
            except ValueError:
                continue
    return by_id


def _gap_evidence(a):
    """Shared evidence view for every gap agent: what was done, found, and admitted as limited."""
    return (f"[{a['ref_id']}] {a['title'][:90]}\n"
            f"   method: {_clip_words(a.get('method', ''), 30)}\n"
            f"   results: {'; '.join(_clip_words(str(x), 25) for x in (a.get('key_results') or [])[:3])}\n"
            f"   metrics: {'; '.join(map(str, (a.get('metrics') or [])[:4]))}\n"
            f"   limitations: {'; '.join(map(str, (a.get('limitations') or [])[:4]))}\n"
            f"   contributions: {'; '.join(_clip_words(str(x), 20) for x in (a.get('contributions') or [])[:3])}")


def verify_gaps(topic, gaps, reviews, analyses):
    ev = "\n".join(_gap_evidence(a) for a in analyses)
    items = []
    for i, g in enumerate(gaps):
        r = reviews.get(i, {})
        items.append(f"G{i}: {g['gap']}\n   papers: {g.get('supporting_papers')} | skeptic: "
                     f"{r.get('verdict', 'n/a')} - {r.get('issue', '')} | revision: {r.get('revised_gap', '')}")
    prompt = f"""You are the Grounded Verifier. Make the final decision for each candidate research gap
on "{topic}", weighing the Skeptic's critique against the paper limitation evidence.
Keep a gap only if the evidence genuinely supports it. Use the Skeptic's revision when it is better.

LIMITATION EVIDENCE:
{ev}

CANDIDATES:
{chr(10).join(items)}

Return JSON {{"verified_gaps": [{{"id": "G0", "gap": "final wording", "supporting_papers": [ids],
"evidence_basis": "which limitations support it", "confidence": 0.0-1.0}}],
"rejected": [{{"id": "G1", "reason": "..."}}]}}"""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="gap_verifier", max_tokens=1300, default={})
    return out if isinstance(out, dict) else {}


def research_gap_triad(topic, analyses, cross, papers, adversarial=True):
    proposed = propose_gaps(topic, analyses, cross)
    if not adversarial:
        return {"proposed": proposed, "final": proposed, "mode": "proposer_only"}
    gated, gate_failed = evidence_gate(proposed, analyses, papers)
    reviews = skeptic_review(topic, gated, analyses) if gated else {}
    verdict = verify_gaps(topic, gated, reviews, analyses) if gated else {}
    final = [g for g in verdict.get("verified_gaps", []) if isinstance(g, dict) and g.get("gap")]
    return {
        "proposed": proposed,
        "gate_failed": gate_failed,
        "gated": gated,
        "skeptic_reviews": {f"G{k}": v for k, v in reviews.items()},
        "final": final,
        "rejected": verdict.get("rejected", []),
        "mode": "adversarial_triad",
    }


# ---------------------------------------------------------------------------
# 8. Taxonomy (deterministic clustering)
# ---------------------------------------------------------------------------
def taxonomy(analyses):
    from sklearn.cluster import AgglomerativeClustering
    if len(analyses) < 3:
        return [{"papers": [a["ref_id"] for a in analyses]}]
    texts = [f"{a['title']}. {a.get('problem', '')} {a.get('method', '')}" for a in analyses]
    v = embedder().encode(texts, normalize_embeddings=True)
    k = 2 if len(analyses) <= 5 else 3
    labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(v)
    groups = {}
    for a, l in zip(analyses, labels):
        groups.setdefault(int(l), []).append(a["ref_id"])
    return [{"papers": g} for g in groups.values()]


# ---------------------------------------------------------------------------
# 9. Citation structuring
# ---------------------------------------------------------------------------
def ieee_reference(p):
    authors = p.get("authors") or []
    a = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    venue = p.get("venue") or p.get("source", "")
    doi = f" doi: {p['doi']}." if p.get("doi") else f" [Online]. Available: {p.get('url')}"
    return f"[{p['ref_id']}] {a}, \"{p['title']},\" {venue}, {p.get('year')}.{doi}"


def references_block(papers):
    return "## References\n\n" + "\n\n".join(ieee_reference(p) for p in papers)


# ---------------------------------------------------------------------------
# 10. Citation-grounded writer (same instructions for every configuration)
# ---------------------------------------------------------------------------
WRITER_RULES = """Write an academic literature review (900-1300 words) in Markdown.
Structure:
## Introduction
2-3 thematic sections (## <theme name>) that synthesize the work
## Comparative Analysis
## Research Gaps and Future Directions
## Conclusion
Rules:
- Support every factual statement with numeric citations like [1] or [2, 3].
- Synthesize across papers (compare, contrast, relate); do not describe papers one by one.
- Report concrete methods, datasets and numbers where available.
- Write in prose paragraphs only (no tables, no bullet lists).
- Do not overstate certainty."""


def write_review(topic, mode, papers=None, analyses=None, cross=None, gaps=None, groups=None,
                 model=None, prompt_variant=None, preferences=""):
    """Writer agent. `model`, `prompt_variant` and `preferences` are only set by the baseline,
    prompt-comparison and user-preference experiments; the ablation configs leave them unset."""
    if mode == "vanilla":
        prompt = f"""Topic: "{topic}"
{WRITER_RULES}
- You have no provided sources: cite real published papers you know and end with
  a "## References" section listing them as [n] Authors, "Title," Venue, Year."""
        return llm.chat([{"role": "user", "content": prompt}], role="writer", model=model, max_tokens=2600)

    if mode == "retrieval":
        src = "\n\n".join(
            f"[{p['ref_id']}] {p['title']} ({p.get('year')}, {p.get('venue', '')})\n"
            f"Abstract: {_clip_words(p['abstract'], 220)}" for p in papers)
        prompt = f"""Topic: "{topic}"
SOURCES (cite only these, by number):
{src}

{WRITER_RULES}
- Cite only the numbered sources above. Do not write a reference list."""
        return llm.chat([{"role": "user", "content": prompt}], role="writer", max_tokens=2600)

    # evidence-grounded modes
    src = "\n\n".join(compact_analysis(a) for a in analyses)
    blocks = [f'Topic: "{topic}"', "SOURCES WITH EXTRACTED EVIDENCE (cite only these, by number):", src]
    if groups:
        blocks.append("PAPER GROUPS (by topical similarity): " +
                      "; ".join(str(g["papers"]) for g in groups))
    if cross:
        blocks.append("CROSS-PAPER ANALYSIS (use this to compare and contrast):\n" + compact_cross(cross))
    if gaps:
        gl = "\n".join(f"- {g.get('gap')} {g.get('supporting_papers', [])}" for g in gaps[:6])
        blocks.append("RESEARCH GAPS (use these in the gap section, cite their supporting papers):\n" + gl)
    if prompt_variant:
        from malrg.prompts import PROMPTS, extract_review
        text = PROMPTS[prompt_variant]["template"].format(context="\n\n".join(blocks),
                                                          preferences=preferences or "")
        out = llm.chat([{"role": "user", "content": text}], role="writer", model=model, max_tokens=3200)
        return extract_review(out, prompt_variant)
    blocks.append(WRITER_RULES + "\n- Cite only the numbered sources above. Do not write a reference list."
                  + (preferences or ""))
    return llm.chat([{"role": "user", "content": "\n\n".join(blocks)}], role="writer", model=model,
                    max_tokens=2600)


def clean_review(md):
    """Strip a model-written reference list (the citation agent appends the verified one)."""
    out = []
    for heading, body in split_markdown_sections(md):
        if re.search(r"reference|bibliograph", heading, re.I):
            break
        out.append((heading, body))
    lines = []
    for h, b in out:
        h = re.sub(r"[*_`]+", "", h).strip()
        if h != "Preamble":
            lines.append(f"## {h}" if not h.lower().startswith("literature review") else f"# {h}")
        if b:
            lines.append(_normalize_prose(b))
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_prose(block):
    """Flatten markdown emphasis / list markers so every sentence is plain, verifiable text.
    Applied identically to every configuration."""
    paras, cur = [], []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        if s.startswith("|") or re.match(r"^[-*_]{3,}$", s):
            continue
        s = re.sub(r"^[-*+]\s+|^\d+\.\s+", "", s)
        s = re.sub(r"[*`]+", "", s)
        cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paras)


# ---------------------------------------------------------------------------
# 11. Verification-Guided Iterative Refinement (feedback loop)
# ---------------------------------------------------------------------------
CERTAINTY_RE = re.compile(r"\b(definitively|conclusively|unequivocally|proves?|proven|always|"
                          r"guarantees?|undeniabl[ey]|clearly demonstrates?)\b", re.I)
CLAIM_RE = re.compile(r"\d|%|\b(achiev|report|propos|demonstrat|outperform|show|found|improv|"
                      r"reduc|increas|accura|use[sd]?|employ|introduc|evaluat)", re.I)


def critic(review_md, index, verifier, n_papers):
    """Local (no-LLM) critic. Returns per-sentence diagnostics and coverage gaps."""
    from evaluation.nli import best_support_any, verify_claim
    issues = []
    cited_refs = set()
    for heading, sent in body_sentences(review_md):
        refs = [r for r in parse_citations(sent) if 1 <= r <= n_papers]
        cited_refs.update(refs)
        claim = strip_citations(sent)
        if refs:
            res = verify_claim(claim, refs, index, verifier)
            if not res["supported"]:
                alt = best_support_any(claim, index, verifier)
                issues.append({"sentence": sent, "heading": heading, "type": "unsupported",
                               "refs": refs, "entail": res["entail"],
                               "alt": alt if alt and alt["entail"] >= config.NLI_ENTAIL_THRESHOLD else None,
                               "evidence": res.get("evidence", "")})
        elif CLAIM_RE.search(claim) and not re.search(r"\bthis review\b|\bwe\b", claim, re.I):
            alt = best_support_any(claim, index, verifier)
            issues.append({"sentence": sent, "heading": heading, "type": "uncited",
                           "alt": alt if alt and alt["entail"] >= config.NLI_ENTAIL_THRESHOLD else None,
                           "evidence": alt["passage"] if alt else ""})
        if CERTAINTY_RE.search(claim):
            issues.append({"sentence": sent, "heading": heading, "type": "overclaim"})
    uncovered = [r for r in range(1, n_papers + 1) if r not in cited_refs]
    return issues, uncovered


def _replace_once(md, old, new):
    i = md.find(old)
    if i == -1:
        return md, False
    return md[:i] + new + md[i + len(old):], True


def _set_citation(sentence, ref):
    s = re.sub(r"\s*\[\d+(?:\s*[,–\-]\s*\d+)*\]", "", sentence).rstrip()
    end = s[-1] if s and s[-1] in ".!?" else ""
    body = s[:-1] if end else s
    return f"{body} [{ref}]{end or '.'}"


def _evidence_for(it, index):
    """Evidence sentences for a flagged sentence: from its cited papers first, then the best papers overall."""
    claim = strip_citations(it["sentence"])
    refs = it.get("refs") or []
    ev = index.top_sentences(claim, refs, k=2) if refs else []
    for r, s in index.top_sentences(claim, None, k=3):
        if len(ev) >= 3:
            break
        if (r, s) not in ev:
            ev.append((r, s))
    return ev


def _revise_batch(topic, batch, cov):
    blocks = []
    for k, (it, ev) in enumerate(batch):
        kind = {"unsupported": "claim not supported by its citation",
                "uncited": "factual claim without citation",
                "overclaim": "overstated certainty"}[it["type"]]
        evs = "\n".join(f"   E{k}.{j} [{r}] \"{_clip_words(s, 55)}\"" for j, (r, s) in enumerate(ev))
        blocks.append(f"S{k} ({kind}): \"{strip_citations(it['sentence'])}\"\n{evs}")
    prompt = f"""You are the Reviser in a verification feedback loop for a literature review on "{topic}".
An NLI verifier found sentences that are not supported by the cited papers. Each flagged sentence
comes with evidence sentences copied from the source papers ([n] = source number).

FLAGGED SENTENCES:
{chr(10).join(blocks) if blocks else "(none)"}

UNCITED SOURCES (each must be mentioned once):
{chr(10).join(cov) if cov else "(none)"}

Instructions:
- For each S#: rewrite the sentence as a faithful paraphrase of the evidence sentence(s) that best match
  its intent. Keep technical terms and numbers exactly as written in the evidence, add nothing that the
  evidence does not state, keep it fluent and suitable for the same paragraph, and end with the citation
  number(s) of the evidence used, e.g. [2]. Return "" only if no evidence sentence is related at all.
- For each U#: write ONE sentence that faithfully paraphrases the given evidence of that source, cites it,
  and name the section (from the listed sections) where it belongs.
Return JSON {{"revisions": [{{"id": "S0", "text": "..."}}],
"additions": [{{"id": "U3", "section": "exact section name", "text": "..."}}]}}"""
    return llm.chat_json([{"role": "user", "content": prompt}], role="reviser", max_tokens=2400,
                         default={}) or {}


def _citation_counts(md, n):
    counts = {r: 0 for r in range(1, n + 1)}
    for _, s in body_sentences(md):
        for r in set(parse_citations(s)):
            if r in counts:
                counts[r] += 1
    return counts


def refine_review(topic, review_md, papers, analyses, index, verifier, log=print):
    """Verification-Guided Iterative Refinement:
    NLI critic -> deterministic re-citation -> evidence-anchored batched reviser ->
    verified-acceptance guard, under a deletion budget that protects source coverage."""
    from evaluation.nli import verify_claim
    n = len(papers)
    history = []
    md = review_md
    draft_sentences = len(body_sentences(review_md))
    deletion_budget = int(config.FEEDBACK_MAX_DELETE_FRACTION * draft_sentences)
    deleted_total = 0
    for rnd in range(1, config.FEEDBACK_MAX_ROUNDS + 1):
        issues, uncovered = critic(md, index, verifier, n)
        n_unsup = sum(1 for i in issues if i["type"] in ("unsupported", "uncited"))
        history.append({"round": rnd, "issues": len(issues), "unsupported_or_uncited": n_unsup,
                        "uncovered_sources": uncovered})
        log(f"    [feedback] round {rnd}: {n_unsup} unsupported/uncited, "
            f"{sum(i['type'] == 'overclaim' for i in issues)} overclaims, uncovered={uncovered}")
        if not issues and not uncovered:
            break

        # (a) deterministic citation repair: another source entails the claim
        llm_items, repaired = [], 0
        for it in issues:
            if it["type"] in ("unsupported", "uncited") and it.get("alt"):
                md, ok = _replace_once(md, it["sentence"], _set_citation(it["sentence"], it["alt"]["ref"]))
                repaired += ok
            else:
                llm_items.append(it)
        llm_items = llm_items[:config.FEEDBACK_MAX_SENTENCES_PER_ROUND]
        prepared = [(it, _evidence_for(it, index)) for it in llm_items]

        cov = []
        sections = [h for h, _ in split_markdown_sections(md) if h != "Preamble"]
        for r in uncovered[:3]:
            p = next(pp for pp in papers if pp["ref_id"] == r)
            ev = index.top_sentences(f"{topic}. {p['title']}", [r], k=3)
            evs = " | ".join(f"\"{_clip_words(s, 45)}\"" for _, s in ev)
            cov.append(f"U{r}: source [{r}] \"{p['title']}\" is never cited. Evidence: {evs}\n"
                       f"   sections: {sections}")

        stats = {"deterministic_recitations": repaired, "llm_accepted": 0, "llm_deleted": 0,
                 "llm_rejected_by_guard": 0, "deletions_refused": 0, "coverage_additions": 0,
                 "llm_items": len(prepared)}
        batch_size = config.FEEDBACK_BATCH_SIZE
        batches = [prepared[i:i + batch_size] for i in range(0, len(prepared), batch_size)] or [[]]
        for b_idx, batch in enumerate(batches):
            batch_cov = cov if b_idx == 0 else []
            if not batch and not batch_cov:
                continue
            out = _revise_batch(topic, batch, batch_cov)
            for rv in out.get("revisions", []) or []:
                try:
                    it, _ = batch[int(str(rv.get("id", "S-1"))[1:])]
                except (ValueError, IndexError):
                    continue
                text = (rv.get("text") or "").strip()
                if not text:
                    counts = _citation_counts(md, n)
                    sole = any(counts.get(r, 0) <= 1 for r in parse_citations(it["sentence"]))
                    if deleted_total >= deletion_budget or sole:
                        stats["deletions_refused"] += 1      # protect content and coverage
                        continue
                    md, ok = _replace_once(md, it["sentence"], "")
                    stats["llm_deleted"] += ok
                    deleted_total += ok
                    continue
                new_refs = [r for r in parse_citations(text) if 1 <= r <= n]
                verified = bool(new_refs) and verify_claim(strip_citations(text), new_refs, index,
                                                           verifier)["supported"]
                if verified or (it["type"] == "overclaim" and not CERTAINTY_RE.search(text)
                                and new_refs):
                    md, ok = _replace_once(md, it["sentence"], text)
                    stats["llm_accepted"] += ok
                else:
                    stats["llm_rejected_by_guard"] += 1   # keep the original sentence
            for ad in out.get("additions", []) or []:
                text = (ad.get("text") or "").strip()
                refs = [r for r in parse_citations(text) if 1 <= r <= n]
                if not text or not refs:
                    continue
                if not verify_claim(strip_citations(text), refs, index, verifier)["supported"]:
                    continue
                md = _insert_into_section(md, ad.get("section", ""), text)
                stats["coverage_additions"] += 1
        history[-1].update(stats)
        md = re.sub(r"[ \t]{2,}", " ", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
    final_issues, final_uncovered = critic(md, index, verifier, n)
    history.append({"round": "final",
                    "unsupported_or_uncited": sum(1 for i in final_issues
                                                  if i["type"] in ("unsupported", "uncited")),
                    "uncovered_sources": final_uncovered,
                    "sentences_draft": draft_sentences, "sentences_final": len(body_sentences(md))})
    return md, history


def _insert_into_section(md, section, text):
    lines = md.split("\n")
    target = None
    for i, line in enumerate(lines):
        if line.startswith("#") and section and section.lower() in line.lower():
            target = i
            break
    if target is None:
        for i, line in enumerate(lines):
            if line.startswith("#") and "comparative" in line.lower():
                target = i
                break
    if target is None:
        return md + "\n\n" + text
    # end of that section's first paragraph block
    j = target + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
        j += 1
    lines[j - 1] = lines[j - 1].rstrip() + " " + text
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional human feedback (UI hook): apply free-text reviewer comments
# ---------------------------------------------------------------------------
def apply_user_feedback(topic, review_md, feedback, analyses, index, verifier):
    """Revise a review according to user comments, then re-run the verification loop."""
    src = "\n".join(compact_analysis(a, with_quotes=False) for a in analyses)
    prompt = f"""Revise this literature review on "{topic}" according to the reviewer feedback.
Keep numeric citations and use only the sources listed.

REVIEWER FEEDBACK:
{feedback}

SOURCES:
{src}

REVIEW:
{review_md}

Return the full revised review in Markdown."""
    revised = llm.chat([{"role": "user", "content": prompt}], role="reviser", max_tokens=2600)
    return revised
