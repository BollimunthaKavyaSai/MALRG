"""
Extended evaluation metrics (all local, no LLM calls, no human labels).

  soft_groundedness        mean NLI entailment probability of cited sentences (SummaC-style soft score)
  numeric_fidelity         share of reported numbers (%, decimals) that occur in the source full text
  semantic_alignment       mean max cosine similarity of review sentences to source passages (MiniLM)
  source_content_recall    share of each source's abstract sentences that some review sentence covers
  topic_relevance          cosine similarity between the review and the research topic/intent
  citation_density         citations per 100 words
  lexical_diversity        MTLD (measure of textual lexical diversity)
  readability_fk_grade     Flesch-Kincaid grade level (lower = easier to read)
  flesch_reading_ease      Flesch reading ease (higher = easier to read)
"""

import re

import numpy as np

from malrg.retrieval import embedder
from malrg.text_utils import body_sentences, parse_citations, split_sentences, strip_citations

NUM_RE = re.compile(r"(?<![\w\[])(\d+(?:\.\d+)?)\s?(%|percent)?(?![\w\]])")


def _numbers(sentence):
    """Result-like numbers: percentages and decimals; skips years, citations and small integers."""
    s = re.sub(r"\[[\d,\s\-–]+\]", " ", sentence)
    out = []
    for val, pct in NUM_RE.findall(s):
        if pct or "." in val:
            if not pct and re.fullmatch(r"(19|20)\d\d", val):
                continue
            out.append(val.rstrip("0").rstrip(".") if "." in val else val)
    return out


def numeric_fidelity(md, papers):
    texts = {p["ref_id"]: " ".join(p["passages"]) + " " + p.get("abstract", "") for p in papers}
    all_text = " ".join(texts.values())
    found = total = 0
    for _, s in body_sentences(md):
        nums = _numbers(s)
        if not nums:
            continue
        refs = [r for r in parse_citations(s) if r in texts]
        pool = " ".join(texts[r] for r in refs) if refs else all_text
        for n in nums:
            total += 1
            if re.search(rf"(?<![\d.]){re.escape(n)}(?:0*)(?![\d])", pool):
                found += 1
    return found / total if total else np.nan, total


def _syllables(word):
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def readability(text):
    sents = split_sentences(text)
    words = re.findall(r"[A-Za-z]+", text)
    if not sents or not words:
        return np.nan, np.nan
    syl = sum(_syllables(w) for w in words)
    wps = len(words) / len(sents)
    spw = syl / len(words)
    fk_grade = 0.39 * wps + 11.8 * spw - 15.59
    ease = 206.835 - 1.015 * wps - 84.6 * spw
    return fk_grade, ease


def mtld(text, threshold=0.72):
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", text)]
    if len(toks) < 50:
        return np.nan

    def one_pass(seq):
        factors, types, count = 0.0, set(), 0
        for t in seq:
            count += 1
            types.add(t)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            factors += (1 - len(types) / count) / (1 - threshold)
        return len(seq) / factors if factors else np.nan

    return float(np.nanmean([one_pass(toks), one_pass(toks[::-1])]))


_SOURCE_CACHE = {}


def _source_vectors(bench):
    """Passage and abstract-sentence embeddings of a topic's sources, computed once per topic
    (they are identical for every configuration)."""
    key = bench["slug"]
    if key not in _SOURCE_CACHE:
        emb = embedder()
        papers = bench["papers"]
        passages = [t for p in papers for t in p["passages"]] + [p.get("abstract", "") for p in papers]
        abs_sents = [s for p in papers for s in split_sentences(p.get("abstract", "")) if len(s.split()) >= 6]
        topic_text = f"{bench['topic']}. {bench.get('intent', {}).get('research_intent', '')}"
        _SOURCE_CACHE[key] = (
            emb.encode(passages, normalize_embeddings=True, batch_size=64),
            emb.encode(abs_sents, normalize_embeddings=True, batch_size=64) if abs_sents else None,
            emb.encode([topic_text], normalize_embeddings=True)[0],
        )
    return _SOURCE_CACHE[key]


def extended_metrics(run, bench, eval_rows):
    md = run["review_md"]
    papers = bench["papers"]
    sents = [s for _, s in body_sentences(md)]
    plain = " ".join(strip_citations(s) for s in sents)
    out = {}

    cited = [r for r in eval_rows if r.get("citations") and "entail" in r]
    out["soft_groundedness"] = float(np.mean([r["entail"] for r in cited])) if cited else 0.0

    fid, n_nums = numeric_fidelity(md, papers)
    out["numeric_fidelity"] = fid
    out["numbers_reported"] = n_nums

    emb = embedder()
    if sents:
        sv = emb.encode([strip_citations(s) for s in sents], normalize_embeddings=True, batch_size=64)
        pv, av, tv = _source_vectors(bench)
        out["semantic_alignment"] = float((sv @ pv.T).max(axis=1).mean())
        if av is not None:
            out["source_content_recall"] = float(((av @ sv.T).max(axis=1) >= 0.6).mean())
        dv = emb.encode([plain[:4000]], normalize_embeddings=True)[0]
        out["topic_relevance"] = float(dv @ tv)
    words = len(re.findall(r"\b\w+\b", plain))
    n_cites = sum(len(parse_citations(s)) for s in sents)
    out["citation_density"] = 100 * n_cites / words if words else 0.0
    out["lexical_diversity"] = mtld(plain)
    fk, ease = readability(plain)
    out["readability_fk_grade"] = fk
    out["flesch_reading_ease"] = ease
    return out
