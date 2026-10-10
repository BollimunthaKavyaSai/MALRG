"""
Minimal OpenAlex client (no API key needed) with an on-disk cache.

Used for (1) building the frozen benchmark paper sets and (2) verifying that
references written by a model correspond to real publications.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from grounding import normalize_title, title_similarity

from .config import BENCHMARK_DIR

API = "https://api.openalex.org/works"
CACHE_PATH = BENCHMARK_DIR / "_openalex_cache.json"
_LOCK = threading.Lock()
_cache: Optional[Dict[str, Any]] = None


# Loads the saved OpenAlex answers (a JSON file) the first time; later calls reuse the copy in memory.
def _load_cache() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


# Writes the OpenAlex answer cache to its JSON file (with a lock, so threads do not collide).
def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")


# HTTP GET to OpenAlex/Europe PMC with a disk cache and up to 4 retries.
# If the same URL was asked before, the saved answer is returned without using the internet.
# On failure it waits 2, 4, 6 ... seconds and tries again; after the last try it raises an error.
def _get(url: str, retries: int = 4) -> Dict[str, Any]:
    cache = _load_cache()
    if url in cache:
        return cache[url]
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MALRG-benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=40) as response:
                payload = json.loads(response.read().decode("utf-8"))
            cache[url] = payload
            _save_cache()
            time.sleep(0.15)
            return payload
        except Exception as error:  # network hiccups / 429
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenAlex request failed: {last}")


# OpenAlex stores an abstract as {word: [positions]}. This puts the words back in position order
# to rebuild the readable abstract text.
def reconstruct_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
    if not inverted:
        return ""
    positions = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    return " ".join(word for _, word in sorted(positions))


# Converts an OpenAlex 'work' into the project's paper dictionary: id, title, authors, year, abstract, DOI,
# links, all PDF links, venue, PubMed Central id, citation count and type.
def work_to_paper(work: Dict[str, Any]) -> Dict[str, Any]:
    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    source = (primary.get("source") or {}) if isinstance(primary, dict) else {}
    doi = str(work.get("doi") or "").replace("https://doi.org/", "")
    return {
        "paper_id": work.get("id"),
        "title": (work.get("title") or work.get("display_name") or "").strip(),
        "authors": [
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ],
        "year": work.get("publication_year"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "doi": doi,
        "url": work.get("doi") or work.get("id"),
        "pdf_url": (best or {}).get("pdf_url") or (primary or {}).get("pdf_url") or "",
        "pdf_urls": list(dict.fromkeys(
            loc.get("pdf_url") for loc in (work.get("locations") or [])
            if isinstance(loc, dict) and loc.get("pdf_url")
        )),
        "venue": source.get("display_name", ""),
        "pmcid": str((work.get("ids") or {}).get("pmcid") or "").rstrip("/").split("/")[-1],
        "cited_by_count": work.get("cited_by_count", 0),
        "type": work.get("type"),
        "source": "OpenAlex",
    }


# Searches OpenAlex for works matching a query, limited to the years, with an abstract, type article/review
# and (optionally) open access. Returns the results converted to paper dictionaries.
def search_works(query: str, start_year: int, end_year: int, per_page: int = 50, open_access: bool = True) -> List[Dict[str, Any]]:
    filters = [
        f"from_publication_date:{start_year}-01-01",
        f"to_publication_date:{end_year}-12-31",
        "has_abstract:true",
        "type:article|review",
    ]
    if open_access:
        filters.append("is_oa:true")
    params = {
        "search": query,
        "filter": ",".join(filters),
        "per-page": str(per_page),
    }
    payload = _get(f"{API}?{urllib.parse.urlencode(params)}")
    return [work_to_paper(w) for w in payload.get("results", [])]


# Finds the real OpenAlex paper that matches a reference title (used to check that a reference exists).
# Searches by title and accepts the best result only if its similarity is above the threshold (0.85).
def find_title(title: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
    """Return the best-matching OpenAlex work for a reference title, if it exists."""
    norm = normalize_title(title)
    if len(norm.split()) < 3:
        return None
    params = {"search": norm[:250], "per-page": "5"}
    try:
        payload = _get(f"{API}?{urllib.parse.urlencode(params)}")
    except RuntimeError:
        return None
    best, best_score = None, 0.0
    for work in payload.get("results", []):
        score = title_similarity(title, work.get("title") or "")
        if score > best_score:
            best, best_score = work, score
    if best is not None and best_score >= threshold:
        paper = work_to_paper(best)
        paper["match_score"] = round(best_score, 3)
        return paper
    return None


# Looks up the PubMed Central id of an open-access article by its DOI using the Europe PMC search API.
def pmcid_for_doi(doi: str) -> str:
    """Look up the PubMed Central id of a DOI through the Europe PMC search API."""
    doi = (doi or "").strip()
    if not doi:
        return ""
    params = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"})
    try:
        payload = _get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}")
    except RuntimeError:
        return ""
    for row in (payload.get("resultList") or {}).get("result", []):
        if row.get("pmcid") and row.get("isOpenAccess", "N") == "Y":
            return row["pmcid"]
    return ""


# Downloads the free full text (XML) of an open-access article from Europe PMC and converts it to plain text.
def europe_pmc_fulltext(pmcid: str) -> str:
    """Full text of an open-access PMC article via the official Europe PMC REST API."""
    import re as _re

    pmcid = (pmcid or "").strip()
    if not pmcid:
        return ""
    if not pmcid.upper().startswith("PMC"):
        pmcid = "PMC" + pmcid
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MALRG-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            xml = response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    xml = _re.sub(r"(?s)<ref-list.*?</ref-list>", " ", xml)
    xml = _re.sub(r"(?s)<(table-wrap|fig|graphic|xref)[^>]*>.*?</\1>", " ", xml)
    xml = _re.sub(r"</(p|title|sec)>", "\n", xml)
    text = _re.sub(r"<[^>]+>", " ", xml)
    text = _re.sub(r"[ \t]+", " ", text)
    return _re.sub(r"\n\s*\n+", "\n\n", text).strip()


# Checks a DOI through OpenAlex and accepts it only if the paper's title broadly agrees with the given title.
def find_doi(doi: str, title: str, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """Resolve a DOI through OpenAlex; accept it only if the title broadly agrees."""
    doi = (doi or "").strip().rstrip(".").lower()
    if not doi.startswith("10."):
        return None
    try:
        work = _get(f"{API}/doi:{urllib.parse.quote(doi)}")
    except RuntimeError:
        return None
    if not work or not work.get("title"):
        return None
    score = title_similarity(title, work.get("title") or "")
    if score >= threshold:
        paper = work_to_paper(work)
        paper["match_score"] = round(score, 3)
        return paper
    return None
