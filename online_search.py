"""
Online fallback for the Search Agent.

When the MongoDB corpus does not contain enough relevant papers for a topic, the Search Agent
calls fetch_and_store() to search OpenAlex (a free, open index of scholarly works with an
official API), keeps only peer-reviewed, open-access papers with a non-preprint PDF inside the
requested year range, and stores them in MongoDB through the Express API. The next corpus search
then finds them, so they are ranked exactly like the papers that were already in the database,
and they stay in the database for future reviews.
"""

import os
import time
from typing import Any, Dict, List

import requests

from populate_corpus import (
    clean,
    get_abstract,
    get_pdf,
    get_publication_type,
    is_preprint,
    normalize_doi,
)

OPENALEX_URL = "https://api.openalex.org/works"
CORPUS_URL = os.getenv("MALRG_CORPUS_URL", "http://127.0.0.1:3000/api/corpus/papers")
# OpenAlex asks API users to identify themselves with a contact e-mail ("polite pool").
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")

RESULTS_PER_QUERY = 25        # OpenAlex records requested per search query
MAX_QUERIES = 4               # search queries sent to OpenAlex per review
REQUEST_DELAY_SECONDS = 0.3   # pause between OpenAlex requests
PEER_REVIEWED_TYPES = "article|proceedings-article"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MALRG/1.0 (multi-agent literature review generator)"})


# Asks OpenAlex (a free database of scholarly papers) for papers matching ONE search query.
# Filters in the request: published between start_year and end_year, type article or proceedings-article
# (peer reviewed), and open access (so a free PDF exists). Returns up to 25 raw results.
def _openalex_search(query: str, start_year: int, end_year: int) -> List[Dict[str, Any]]:
    params = {
        "search": query,
        "filter": (
            f"from_publication_date:{start_year}-01-01,"
            f"to_publication_date:{end_year}-12-31,"
            f"type:{PEER_REVIEWED_TYPES},"
            "is_oa:true"
        ),
        "per-page": RESULTS_PER_QUERY,
    }
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    response = SESSION.get(OPENALEX_URL, params=params, timeout=45)
    response.raise_for_status()
    return response.json().get("results", [])


# Converts one OpenAlex result into the paper format that the project stores in MongoDB.
# It is rejected (returns None) when: the title or year is missing, the year is outside the range,
# there is no real PDF link, the PDF is a preprint, or the abstract is shorter than 30 words.
# Otherwise it copies title, authors, year, abstract, DOI, venue, links and marks the source
# as 'OpenAlex (online fallback)'.
def _to_paper(work: Dict[str, Any], query: str, start_year: int, end_year: int):
    """Convert an OpenAlex work into the MongoDB paper schema, or None if it is not usable."""
    year = work.get("publication_year")
    title = clean(work.get("title"))
    if not title or not isinstance(year, int) or not start_year <= year <= end_year:
        return None
    pdf_url = get_pdf(work)
    if not pdf_url or is_preprint(work, pdf_url):
        return None
    abstract = get_abstract(work)
    if len(abstract.split()) < 30:          # ranking and analysis need a real abstract
        return None

    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    authors = [
        clean((a.get("author") or {}).get("display_name"))
        for a in work.get("authorships") or [] if isinstance(a, dict)
    ]
    # same short id format as the papers already in the corpus (e.g. "W4387682130")
    paper_id = clean(work.get("id")).rsplit("/", 1)[-1] or normalize_doi(work.get("doi"))
    if not paper_id:
        return None

    # Only metadata is sent: if the paper already exists, its stored evidence fields are kept.
    return {
        "paper_id": paper_id,
        "title": title,
        "authors": [a for a in authors if a],
        "year": year,
        "publication_date": work.get("publication_date"),
        "abstract": abstract,
        "doi": normalize_doi(work.get("doi")),
        "venue": clean(source.get("display_name")),
        "publication_types": [get_publication_type(work)],
        "url": clean(primary.get("landing_page_url")),
        "pdf_url": pdf_url,
        "source": "OpenAlex (online fallback)",
        "is_open_access": True,
        "is_preprint": False,
        "is_available": True,
        "retrieval_queries": [query],
    }


# Online fallback of the Search agent: when MongoDB does not hold enough relevant papers.
#  1. Searches OpenAlex with up to 4 of the agent's queries (waiting 0.3 s between requests).
#  2. Converts every usable result with _to_paper() and removes duplicates.
#  3. Sends all new papers to the Node endpoint POST /api/corpus/papers, which saves them in MongoDB.
# Returns how many usable papers were found, how many were new, how many already existed, and any errors.
# It never crashes the run: a network problem simply means no extra papers are added.
# Saved papers stay in MongoDB, so future reviews can reuse them.
def fetch_and_store(queries: List[str], start_year: int, end_year: int) -> Dict[str, Any]:
    """Search OpenAlex with the agent's queries and upsert the usable papers into MongoDB.

    Returns {"found": n_usable, "inserted": n_new, "updated": n_existing, "errors": [...]}.
    Never raises: a network problem simply means no online papers are added.
    """
    papers, seen, errors = [], set(), []
    for query in [q for q in queries if q][:MAX_QUERIES]:
        try:
            works = _openalex_search(query, start_year, end_year)
        except requests.RequestException as exc:
            errors.append(f"OpenAlex search failed for '{query}': {exc}")
            continue
        for work in works:
            paper = _to_paper(work, query, start_year, end_year)
            if paper and paper["paper_id"] not in seen:
                seen.add(paper["paper_id"])
                papers.append(paper)
        time.sleep(REQUEST_DELAY_SECONDS)

    result = {"found": len(papers), "inserted": 0, "updated": 0, "errors": errors}
    if not papers:
        return result
    try:
        response = SESSION.post(CORPUS_URL, json={"papers": papers}, timeout=60)
        response.raise_for_status()
        stored = response.json()
        result["inserted"] = stored.get("inserted", 0)
        result["updated"] = stored.get("updated", 0)
    except requests.RequestException as exc:
        errors.append(f"Storing online papers in MongoDB failed: {exc}")
    return result
