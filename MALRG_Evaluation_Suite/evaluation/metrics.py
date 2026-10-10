"""
Automatic, reference-free evaluation metrics (no manual validation).

Grounding / hallucination (NLI, evaluator model != feedback critic model)
  CG   Citation Groundedness      cited sentences entailed by their cited source(s)
  CP   Citation Precision         citations that individually (or jointly) support their sentence
  HR   Hallucination Rate         claim sentences supported by NO passage of any available source
  CR   Contradiction Rate         claim sentences contradicted by their cited source
  UCR  Uncited Claim Rate         factual-looking sentences that carry no citation
  FRR  Fabricated Reference Rate  references that do not resolve to a real paper / valid source id

Coverage / synthesis
  SC   Source Coverage            benchmark papers cited at least once
  MSR  Multi-Source Synthesis     cited sentences that integrate >= 2 distinct sources
  CPC  Cross-Paper Comparison     sentences with a comparative marker AND >= 2 sources
  CB   Citation Balance           normalized entropy of citation counts over the sources

Review-1 slide-28 indicators: synthesis indicators, gap indicators, certainty flags,
structure, word count, redundancy.
"""

import json
import math
import re
import time

import numpy as np
import requests

import config
from evaluation.nli import PassageIndex, best_support_any, get_verifier, verify_claim
from malrg.retrieval import UA, embedder
from malrg.text_utils import (body_sentences, chunk_text, is_gap_heading, normalize_title,
                              parse_citations, reference_section, split_markdown_sections,
                              split_sentences, strip_citations)

COMPARATIVE_RE = re.compile(
    r"\b(whereas|in contrast|by contrast|unlike|compared (?:to|with)|while|similarly|likewise|"
    r"both|outperform\w*|on the other hand|conversely|however|differ\w*|consistent with|"
    r"contradict\w*|in comparison|than)\b", re.I)
SYNTHESIS_TERMS = ["whereas", "in contrast", "compared", "cross-paper", "unlike", "similarly",
                   "both", "however", "while", "conversely", "consistent with", "trade-off",
                   "complement", "converge", "diverge", "together"]
GAP_TERMS = ["gap", "unresolved", "open challenge", "limitation", "unexplored", "lack", "limited",
             "future work", "remains", "under-explored", "underexplored", "scarce", "need for",
             "open question", "insufficient"]
CERTAINTY_TERMS = ["empirically proven", "definitively", "conclusively", "unequivocally",
                   "proves that", "guarantees", "undeniably", "always", "clearly demonstrates"]
CLAIM_RE = re.compile(r"\d|%|\b(achiev|report|propos|demonstrat|outperform|show|found|improv|"
                      r"reduc|increas|accura|employ|introduc|evaluat|use[sd]?|achieved|attain)", re.I)
META_RE = re.compile(r"\b(this review|this survey|this paper|we (review|discuss|survey|organize)|"
                     r"the following section|in this section)\b", re.I)


# ---------------------------------------------------------------------------
# Reference resolution for the vanilla (M0) configuration
# ---------------------------------------------------------------------------
def parse_reference_list(md):
    refs = {}
    block = reference_section(md)
    for line in re.split(r"\n+", block):
        m = re.match(r"^\s*[-*]?\s*\[?(\d+)\]?[.)]?\s+(.*)$", line.strip())
        if m:
            refs[int(m.group(1))] = m.group(2).strip()
    return refs


def _extract_title(ref_text):
    m = re.search(r"[\"“](.+?)[\"”]", ref_text)
    if m:
        return m.group(1).strip(" ,.")
    m = re.search(r"\(\d{4}\)\.?\s*(.+?)\.", ref_text)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in ref_text.split(".") if len(p.split()) >= 4]
    return parts[1] if len(parts) > 1 else (parts[0] if parts else ref_text)


_RESOLVE_CACHE = {}
_RESOLVE_FILE = config.CACHE_DIR.parent / "reference_resolution.json"


def _title_match(key, cand):
    """Jaccard similarity of title tokens; a candidate that contains every token of a 4+-word
    query title (e.g. 'Faculty Opinions recommendation of <title>') also proves existence."""
    ta, tb = set(key.split()), set(normalize_title(cand or "").split())
    if len(ta) >= 4 and ta <= tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def _get(url, params, attempts=4):
    """GET with retry; returns the Response, or None if the service could not be reached."""
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


def _crossref(title, key):
    r = _get("https://api.crossref.org/works",
             {"query.bibliographic": title, "rows": 20, "select": "title,abstract,issued,DOI"})
    if r is None:
        return "failed", None
    try:
        items = r.json().get("message", {}).get("items", [])
    except ValueError:
        return "failed", None
    for w in items:
        cand = (w.get("title") or [""])[0]
        if _title_match(key, cand) >= 0.7:
            abstract = re.sub(r"<[^>]+>", " ", w.get("abstract") or "")
            return "found", {"title": cand, "abstract": normalize_ws_local(abstract), "doi": w.get("DOI"),
                             "source": "Crossref"}
    return "not_found", None


def _dblp(title, key):
    r = _get("https://dblp.org/search/publ/api", {"q": title, "format": "json", "h": 5})
    if r is None:
        return "failed", None
    try:
        hits = ((r.json().get("result") or {}).get("hits") or {}).get("hit") or []
    except ValueError:
        return "failed", None
    for h in hits:
        cand = re.sub(r"\.$", "", (h.get("info") or {}).get("title") or "")
        if _title_match(key, cand) >= 0.7:
            return "found", {"title": cand, "abstract": "", "source": "DBLP"}
    return "not_found", None


def _arxiv(title, key):
    phrase = re.sub(r"[^A-Za-z0-9 \-]", " ", title)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    r = _get("https://export.arxiv.org/api/query", {"search_query": f'ti:"{phrase}"', "max_results": 5})
    if r is None:
        return "failed", None
    for e in re.findall(r"<entry>(.*?)</entry>", r.text, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        cand = normalize_ws_local(t.group(1)) if t else ""
        if _title_match(key, cand) >= 0.7:
            return "found", {"title": cand, "abstract": normalize_ws_local(s.group(1) if s else ""),
                             "source": "arXiv"}
    return "not_found", None


def normalize_ws_local(s):
    return re.sub(r"\s+", " ", s or "").strip()


def resolve_reference(ref_text):
    """Check whether a reference is a real paper (Crossref, then arXiv).

    Returns a dict with status: "found" (real paper; abstract if available),
    "not_found" (both indexes answered and neither has it -> fabricated) or
    "unverified" (a lookup service could not be reached -> excluded, never counted as fabricated).
    Results are cached on disk so re-evaluation is deterministic.
    """
    if not _RESOLVE_CACHE and _RESOLVE_FILE.exists():
        _RESOLVE_CACHE.update(json.loads(_RESOLVE_FILE.read_text(encoding="utf-8")))
    title = _extract_title(ref_text)
    key = normalize_title(title)
    if key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[key]
    if len(key.split()) < 3:
        result = {"status": "unverified", "title": title, "reason": "no parsable title"}
    else:
        statuses, hit = [], None
        for lookup in (_arxiv, _crossref):             # arXiv first (abstracts), then Crossref
            st, h = lookup(title, key)
            statuses.append(st)
            if st == "found":
                hit = h
                break
            time.sleep(0.4)
        if hit is not None:
            status = "found"
        elif "failed" in statuses:
            status = "unverified"                      # could not check every index
        else:
            status = "not_found"                       # all indexes answered: not a real paper
        result = {"status": status, "query_title": title, **(hit or {})}
        time.sleep(0.3)
    if result["status"] != "unverified":          # never cache transient failures
        _RESOLVE_CACHE[key] = result
        _RESOLVE_FILE.write_text(json.dumps(_RESOLVE_CACHE, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _complete_reference(text):
    """A reference cut off by the output-length limit (no year, too short) cannot be judged."""
    return bool(re.search(r"\b(19|20)\d\d\b", text)) and len(text.split()) >= 6


def vanilla_sources(md, bench):
    """Build pseudo-papers for M0's self-generated references (verifiable via OpenAlex)."""
    all_refs = {k: v for k, v in parse_reference_list(md).items() if _complete_reference(v)}
    papers, fabricated, unverifiable, bench_hits = [], [], [], set()
    refs = {}
    bench_titles = {normalize_title(p["title"]): p["ref_id"] for p in bench["papers"]}
    for rid, text in sorted(all_refs.items()):
        res = resolve_reference(text)
        if res["status"] == "unverified":
            unverifiable.append(rid)          # lookup failed: excluded from the rate
            continue
        refs[rid] = text
        if res["status"] == "not_found":
            fabricated.append(rid)
            continue
        if normalize_title(res["title"]) in bench_titles:
            bench_hits.add(bench_titles[normalize_title(res["title"])])
        abstract = res.get("abstract") or ""
        if len(abstract.split()) < 20:
            unverifiable.append(rid)          # real paper, but no text to verify claims against
            continue
        papers.append({"ref_id": rid, "title": res["title"], "abstract": abstract,
                       "passages": chunk_text(abstract, config.CHUNK_WORDS, config.CHUNK_OVERLAP)})
    return refs, papers, fabricated, bench_hits, unverifiable


# ---------------------------------------------------------------------------
# Main per-review evaluation
# ---------------------------------------------------------------------------
def evaluate_review(run, bench, verifier=None):
    verifier = verifier or get_verifier("eval")
    md = run["review_md"]
    is_vanilla = not config.CONFIGS.get(run["config_id"], {}).get("retrieval", True)
    n = len(bench["papers"])

    unverifiable_ids = set()
    if is_vanilla:
        refs, pseudo, fabricated, bench_hits, unverifiable = vanilla_sources(md, bench)
        unverifiable_ids = set(unverifiable)
        cite_index = PassageIndex(pseudo) if pseudo else None
        valid_ids = {p["ref_id"] for p in pseudo}
        n_refs = len(refs)
        # no complete reference list (e.g. output truncated) -> fabrication rate is undefined
        frr = len(fabricated) / n_refs if n_refs else float("nan")
        # generous pool for the "any support" check: its own real refs + the benchmark papers
        offset = 1000
        pool_papers = pseudo + [{**p, "ref_id": p["ref_id"] + offset} for p in bench["papers"]]
        pool_index = PassageIndex(pool_papers)
        covered = bench_hits
    else:
        cite_index = PassageIndex(bench["papers"])
        pool_index = cite_index
        valid_ids = set(range(1, n + 1))
        fabricated = []
        n_refs = n
        covered = set()

    sents = body_sentences(md)
    rows = []
    cite_counts = {}
    invalid_cites = set()
    for heading, s in sents:
        claim = strip_citations(s)
        ids = parse_citations(s)
        good = [i for i in ids if i in valid_ids]
        invalid_cites.update(i for i in ids if i not in valid_ids)
        is_claim = bool(ids) or (bool(CLAIM_RE.search(claim)) and not META_RE.search(claim))
        row = {"heading": heading, "sentence": s, "citations": ids, "is_claim": is_claim,
               "n_sources": len(set(good))}
        if ids and not good and all(i in unverifiable_ids for i in ids):
            row["unverifiable"] = True     # cites only real-but-textless or unreachable references
        for i in good:
            cite_counts[i] = cite_counts.get(i, 0) + 1
            if not is_vanilla:
                covered.add(i)
        if ids:
            if good and cite_index is not None:
                v = verify_claim(claim, good, cite_index, verifier)
            else:
                v = {"supported": False, "entail": 0.0, "contra": 0.0, "contradicted": False,
                     "per_ref_entail": {}}
            row.update({"supported": v["supported"], "entail": v["entail"],
                        "contradicted": v.get("contradicted", False),
                        "per_ref_entail": v.get("per_ref_entail", {})})
        if is_claim:
            if row.get("supported"):
                row["any_support"] = True
            else:
                alt = best_support_any(claim, pool_index, verifier)
                row["any_support"] = bool(alt and alt["entail"] >= config.NLI_ENTAIL_THRESHOLD)
        row["comparative"] = bool(COMPARATIVE_RE.search(claim))
        rows.append(row)

    cited = [r for r in rows if r["citations"] and not r.get("unverifiable")]
    claims = [r for r in rows if r["is_claim"]]
    supported = [r for r in cited if r.get("supported")]

    n_cit_total, n_cit_precise = 0, 0
    for r in cited:
        for c in r["citations"]:
            n_cit_total += 1
            if r.get("supported"):
                pe = r.get("per_ref_entail", {}).get(c, 0.0)
                joint = r["entail"] >= config.NLI_ENTAIL_THRESHOLD and len(r["citations"]) > 1
                if pe >= config.NLI_ENTAIL_THRESHOLD or (joint and pe >= 0.15):
                    n_cit_precise += 1

    text_words = " ".join(s for _, s in sents)
    wc = len(re.findall(r"\b\w+\b", text_words))
    low = text_words.lower()
    gap_text = " ".join(b for h, b in split_markdown_sections(md) if is_gap_heading(h)).lower()

    counts = np.array([cite_counts.get(i, 0) for i in (range(1, n + 1) if not is_vanilla
                                                       else sorted(valid_ids))], dtype=float)
    if counts.sum() > 0 and len(counts) > 1:
        p = counts[counts > 0] / counts.sum()
        balance = float(-(p * np.log(p)).sum() / math.log(len(counts)))
    else:
        balance = 0.0

    m = {
        "citation_groundedness": _ratio(len(supported), len(cited)),
        "citation_precision": _ratio(n_cit_precise, n_cit_total),
        "hallucination_rate": _ratio(sum(1 for r in claims if not r.get("any_support")), len(claims)),
        "contradiction_rate": _ratio(sum(1 for r in cited if r.get("contradicted")), len(claims)),
        "uncited_claim_rate": _ratio(sum(1 for r in claims if not r["citations"]), len(claims)),
        "fabricated_reference_rate": ((len(fabricated) / n_refs if n_refs else float("nan")) if is_vanilla else
                                      _ratio(len(invalid_cites), len(invalid_cites | set(cite_counts)))),
        "source_coverage": len(covered & set(range(1, n + 1))) / n,
        "multi_source_synthesis": _ratio(sum(1 for r in cited if r["n_sources"] >= 2), len(cited)),
        "cross_paper_comparison": _ratio(sum(1 for r in rows if r["comparative"] and r["n_sources"] >= 2),
                                         len(rows)),
        "citation_balance": balance,
        "synthesis_indicators_per_1k": 1000 * sum(low.count(t) for t in SYNTHESIS_TERMS) / max(1, wc),
        "gap_indicators": sum(gap_text.count(t) for t in GAP_TERMS),
        "certainty_flags": sum(low.count(t) for t in CERTAINTY_TERMS),
        "structure_score": structure_score(md),
        "redundancy": redundancy(sents),
        "word_count": wc,
        "n_sentences": len(rows),
        "n_cited_sentences": len(cited),
        "n_claims": len(claims),
        "n_references": n_refs,
        "n_unverifiable_references": len(unverifiable_ids),
        "n_unverifiable_sentences": sum(1 for r in rows if r.get("unverifiable")),
    }
    return m, rows


def _ratio(a, b):
    return a / b if b else 0.0


def structure_score(md):
    heads = [h.lower() for h, _ in split_markdown_sections(md) if h != "Preamble"]
    checks = [
        any("introduction" in h for h in heads),
        any("compar" in h for h in heads),
        any(is_gap_heading(h) for h in heads),
        any("conclusion" in h for h in heads),
        any("reference" in h for h in heads),
        sum(1 for h in heads if not re.search(
            r"introduction|compar|gap|future|conclusion|reference|literature review", h)) >= 2,
    ]
    return sum(checks) / len(checks)


def redundancy(sents):
    texts = [s for _, s in sents]
    if len(texts) < 3:
        return 0.0
    v = embedder().encode(texts, normalize_embeddings=True)
    sim = v @ v.T
    np.fill_diagonal(sim, 0)
    return float((sim.max(axis=1) > 0.9).mean())


# ---------------------------------------------------------------------------
# Pipeline-internal metrics (only for configs that run the corresponding agent)
# ---------------------------------------------------------------------------
FIELDS = ["problem", "dataset", "method", "key_results", "metrics", "limitations", "contributions"]


def extraction_metrics(run):
    an = run.get("analyses") or []
    if not an:
        return {}
    fid = [a.get("evidence_fidelity", 0) for a in an]
    comp = []
    for a in an:
        filled = 0
        for f in FIELDS:
            v = a.get(f)
            txt = " ".join(map(str, v)) if isinstance(v, list) else str(v or "")
            if txt.strip() and "not reported" not in txt.lower():
                filled += 1
        comp.append(filled / len(FIELDS))
    return {"evidence_fidelity": float(np.mean(fid)), "field_completeness": float(np.mean(comp)),
            "verified_quotes": sum(len(a.get("evidence") or []) for a in an)}


def gap_pipeline_metrics(run):
    g = run.get("gap_result")
    if not g:
        return {}
    out = {"gaps_proposed": len(g.get("proposed", [])), "gaps_final": len(g.get("final", []))}
    if g.get("mode") == "adversarial_triad":
        out["gaps_gate_rejected"] = len(g.get("gate_failed", []))
        out["gaps_skeptic_rejected"] = sum(1 for r in g.get("skeptic_reviews", {}).values()
                                           if str(r.get("verdict", "")).lower() == "reject")
        out["gaps_skeptic_revised"] = sum(1 for r in g.get("skeptic_reviews", {}).values()
                                          if str(r.get("verdict", "")).lower() == "revise")
        out["gap_filter_rate"] = 1 - out["gaps_final"] / max(1, out["gaps_proposed"])
    return out


def efficiency_metrics(run):
    u = run.get("usage", {})
    return {"llm_calls": u.get("llm_calls", 0), "total_tokens": u.get("total_tokens", 0),
            "api_latency_s": u.get("api_latency_s", 0.0)}
