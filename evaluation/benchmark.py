"""
Frozen benchmark builder.

For every benchmark topic the same paper set is shared by ALL ablation
configurations, so differences between M0..M5 come only from the agents.

Retrieval mirrors the project's hybrid ranker:
    0.60 semantic (all-MiniLM-L6-v2) + 0.20 TF-IDF + 0.15 title relevance
    + 0.05 metadata quality
over open-access, peer-reviewed OpenAlex works in the year range.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from grounding import get_embedder, normalize_title

from .config import BENCHMARK_DIR, END_YEAR, PAPERS_PER_TOPIC, START_YEAR
from .openalex import search_works

PREPRINT_HOSTS = ("arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "preprints.org", "researchsquare.com")
STOPWORDS = set(
    "a an the of for in on and or to with using based via by from into at as is are towards toward".split()
)


# Turns a topic into a safe lowercase folder name (letters, digits and underscores).
def slugify(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:60]


# Returns the folder benchmark/<slug>/ where all files of one benchmark topic are stored.
def topic_dir(topic: str) -> Path:
    return BENCHMARK_DIR / slugify(topic)


# Returns the meaningful words of a text: normalised, without stop words and very short words.
def _content_words(text: str) -> List[str]:
    return [w for w in normalize_title(text).split() if w not in STOPWORDS and len(w) > 2]


# Ranks candidate papers for a topic with the SAME scoring as the real Search agent:
# 60% MiniLM semantic similarity, 20% TF-IDF, 15% title word match, 5% metadata quality
# (PDF, DOI, venue, citation count). Adds all the scores to each paper and sorts best first.
def rank_candidates(topic: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [f"{p['title']}. {p['abstract']}" for p in candidates]
    embedder = get_embedder()
    topic_vec = embedder.encode([topic], normalize_embeddings=True, show_progress_bar=False)[0]
    doc_vecs = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    semantic = doc_vecs @ topic_vec

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = tfidf.fit_transform(docs + [topic])
    lexical = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    topic_words = set(_content_words(topic))
    title_scores = np.array([
        len(topic_words & set(_content_words(p["title"]))) / max(1, len(topic_words)) for p in candidates
    ])
    quality = np.array([
        0.4 * bool(p.get("pdf_url")) + 0.2 * bool(p.get("doi")) + 0.2 * bool(p.get("venue"))
        + 0.2 * min(1.0, math.log1p(p.get("cited_by_count") or 0) / math.log(200))
        for p in candidates
    ])

    # Rescales scores to 0..1 (lowest 0, highest 1) so different score types can be mixed.
    def norm(x):
        x = np.asarray(x, dtype=float)
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)

    score = 0.60 * norm(semantic) + 0.20 * norm(lexical) + 0.15 * title_scores + 0.05 * quality
    for i, paper in enumerate(candidates):
        paper["semantic_score"] = round(float(semantic[i]), 4)
        paper["tfidf_score"] = round(float(lexical[i]), 4)
        paper["title_score"] = round(float(title_scores[i]), 4)
        paper["metadata_quality_score"] = round(float(quality[i]), 4)
        paper["relevance_score"] = round(float(score[i]), 4)
    return sorted(candidates, key=lambda p: p["relevance_score"], reverse=True)


# Chooses the benchmark papers of one topic from OpenAlex:
# searches open-access works, drops duplicates, preprints and papers with a short abstract,
# ranks them, keeps those with semantic score of at least 0.35 and returns the top n.
def retrieve_papers(topic: str, n: int = PAPERS_PER_TOPIC) -> List[Dict[str, Any]]:
    candidates = search_works(topic, START_YEAR, END_YEAR, per_page=50, open_access=True)
    seen, filtered = set(), []
    for paper in candidates:
        key = normalize_title(paper["title"])
        url = (paper.get("pdf_url") or "") + " " + (paper.get("url") or "")
        if not key or key in seen or len(paper.get("abstract", "")) < 400:
            continue
        if any(host in url for host in PREPRINT_HOSTS):
            continue
        seen.add(key)
        filtered.append(paper)
    ranked = rank_candidates(topic, filtered)
    # Require a minimum semantic fit so off-topic works never enter the set.
    on_topic = [p for p in ranked if p["semantic_score"] >= 0.35] or ranked
    return on_topic[:n]


# If the main PDF could not be read, tries other sources:
# first the free full text from Europe PMC, then every other PDF link that OpenAlex lists for the paper.
def _retry_alternate_pdfs(paper: Dict[str, Any]) -> Dict[str, Any]:
    """If the primary PDF failed, try every other open-access location OpenAlex lists."""
    from paper_extraction import extract_paper_content

    if paper.get("content_source") == "full_text":
        return paper
    from .openalex import europe_pmc_fulltext, pmcid_for_doi

    pmcid = paper.get("pmcid") or pmcid_for_doi(paper.get("doi", ""))
    if pmcid:
        text = europe_pmc_fulltext(pmcid)
        if len(text) > 5000:
            print(f"   ✅ Europe PMC full text ({len(text):,} chars): {paper.get('title', '')[:60]}")
            return {**paper, "content_source": "full_text", "full_text": text, "text_length": len(text)}
    for n, url in enumerate(paper.get("pdf_urls") or []):
        if url == paper.get("pdf_url"):
            continue
        attempt = extract_paper_content({**paper, "pdf_url": url, "paper_id": f"{paper.get('paper_id')}_alt{n}"})
        if attempt.get("content_source") == "full_text":
            attempt["paper_id"] = paper.get("paper_id")
            return attempt
    return paper


# Creates the frozen benchmark files of one topic: papers.json (retrieval), extracted.json (full text) and
# topic.json (statistics). If the files exist they are reused, so every configuration uses identical papers.
def build_topic(topic: str, force: bool = False) -> Dict[str, Any]:
    """Create benchmark/<slug>/papers.json (retrieval) and extracted.json (full text)."""
    from paper_extraction import extract_papers_content

    folder = topic_dir(topic)
    folder.mkdir(parents=True, exist_ok=True)
    papers_path = folder / "papers.json"
    extracted_path = folder / "extracted.json"

    if force or not papers_path.exists():
        papers = retrieve_papers(topic)
        papers_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📚 [{topic}] retrieved {len(papers)} papers")
    papers = json.loads(papers_path.read_text(encoding="utf-8"))

    if force or not extracted_path.exists():
        extracted = extract_papers_content(papers)
        extracted = [_retry_alternate_pdfs(paper) for paper in extracted]
        extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    extracted = json.loads(extracted_path.read_text(encoding="utf-8"))

    meta = {
        "topic": topic,
        "papers": len(papers),
        "full_text_papers": sum(1 for p in extracted if p.get("content_source") == "full_text"),
        "mean_relevance": round(float(np.mean([p["relevance_score"] for p in papers])), 4) if papers else 0,
        "mean_semantic": round(float(np.mean([p["semantic_score"] for p in papers])), 4) if papers else 0,
    }
    (folder / "topic.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
