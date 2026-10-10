import requests
import sys
import time
import re


# ============================================================
# CONFIGURATION
# ============================================================

NODE_URL = "http://127.0.0.1:3000/api/corpus/papers"

# Broad scholarly corpus.
#
# IMPORTANT:
# The corpus must contain papers from multiple research areas.
# The user's actual topic is supplied later during retrieval.
#
# These queries deliberately cover:
#   - multi-agent systems
#   - agentic AI
#   - LLM agents
#   - literature review generation
#   - automated literature reviews
#   - research assistants
#   - scholarly information retrieval
#   - scientific literature synthesis
#   - citation/evidence grounded generation
#   - multi-agent research systems
SEARCH_QUERIES = [
    "multi-agent systems",
    "multi-agent artificial intelligence",
    "multi-agent large language models",
    "LLM agents",
    "agentic AI",
    "AI agents",
    "multi-agent LLM",
    "multi-agent research systems",
    "multi-agent research assistants",
    "literature review generation",
    "automated literature review",
    "automatic literature review",
    "LLM literature review",
    "large language models literature review",
    "AI research assistant",
    "LLM research assistant",
    "scholarly information retrieval",
    "scientific literature retrieval",
    "scientific literature synthesis",
    "automated scientific literature synthesis",
    "citation grounded generation",
    "evidence grounded generation",
    "citation aware language models",
    "scientific question answering",
    "automated scientific discovery",
]

START_YEAR = 2020
END_YEAR = 2026

# Number of unique eligible papers to import.
TARGET = 250

# Maximum OpenAlex records requested per query/type.
PER_PAGE = 100

# Small delay between OpenAlex requests.
REQUEST_DELAY_SECONDS = 0.25


# ============================================================
# PREPRINT FILTERING
# ============================================================

PREPRINT_HOSTS = (
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",
    "engrxiv.org",
    "chemrxiv.org",
    "researchsquare.com",
    "osf.io",
    "preprints.org",
    "zenodo.org",
    "hal.science",
    "researchgate.net",
)


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "MultiAgentLiteratureReviewGenerator/1.0 "
            "(scholarly corpus collection)"
        )
    }
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

# Turns any value into a clean string: None becomes '', extra whitespace is removed.
def clean(value):
    """
    Normalize whitespace and convert values safely to strings.
    """
    return " ".join(str(value or "").split()).strip()


# Prepares text for duplicate detection: lowercase, remove punctuation and extra spaces.
def normalize_text(value):
    """
    Normalize text for duplicate detection.
    """
    text = clean(value).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Makes DOIs comparable by removing 'https://doi.org/' and 'doi:' prefixes and lowercasing,
# so the same paper is not inserted twice.
def normalize_doi(doi):
    """
    Normalize DOI values so the same paper is not inserted twice.
    """
    doi = clean(doi).lower()

    if not doi:
        return ""

    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")

    return doi.strip()


# Rebuilds the readable abstract from OpenAlex's inverted index (words with their positions).
def get_abstract(work):
    """
    Reconstruct OpenAlex inverted-index abstracts.
    """
    inv = work.get("abstract_inverted_index")

    if not isinstance(inv, dict):
        return ""

    words = []

    for word, positions in inv.items():
        if not isinstance(positions, list):
            continue

        for pos in positions:
            try:
                words.append((int(pos), word))
            except (TypeError, ValueError):
                continue

    words.sort(key=lambda x: x[0])

    return " ".join(word for _, word in words)


# ============================================================
# PUBLICATION TYPE
# ============================================================

# Maps the OpenAlex work type (article, proceedings-article, ...) to the project's publication_types field.
def get_publication_type(work):
    """
    Map OpenAlex work types to the project's publication_types field.

    We retain both journal articles and proceedings articles because
    the project is intended to work with published scholarly literature.
    """

    work_type = clean(work.get("type")).lower()

    if work_type == "proceedings-article":
        return "ConferencePaper"

    if work_type == "article":
        return "JournalArticle"

    return "ScholarlyArticle"


# ============================================================
# PDF / SOURCE DISCOVERY
# ============================================================

# True if a URL clearly points to a preprint site (arXiv, bioRxiv, SSRN, ...).
def is_preprint_url(url):
    """
    Check whether a URL clearly points to a known preprint platform.
    """
    url = clean(url).lower()

    if not url:
        return False

    return any(host in url for host in PREPRINT_HOSTS)


# Finds a usable scholarly PDF link among the work's open-access locations, skipping preprint hosts.
def get_pdf(work):
    """
    Find an accessible-looking scholarly PDF URL while avoiding
    obvious preprint hosts.

    OpenAlex can contain several locations for the same paper, so
    inspect all available locations instead of relying on only one.
    """

    locations = []

    best = work.get("best_oa_location") or {}

    if isinstance(best, dict):
        locations.append(best)

    for location in work.get("locations") or []:
        if isinstance(location, dict):
            locations.append(location)

    seen_urls = set()

    for location in locations:

        pdf = clean(location.get("pdf_url"))

        if not pdf:
            continue

        if pdf in seen_urls:
            continue

        seen_urls.add(pdf)

        if is_preprint_url(pdf):
            continue

        return pdf

    return None


# True if the metadata shows the work is a preprint (type, venue or PDF host); such papers are excluded.
def is_preprint(work, pdf_url):
    """
    Reject papers whose available scholarly metadata clearly identifies
    them as coming from an obvious preprint repository.

    We intentionally inspect the title/DOI/landing/PDF metadata rather
    than rejecting papers merely because they are open access.
    """

    primary = work.get("primary_location") or {}

    text_parts = [
        clean(work.get("title")),
        clean(work.get("doi")),
        clean(primary.get("landing_page_url")),
        clean(pdf_url),
    ]

    text = " ".join(text_parts).lower()

    return any(host in text for host in PREPRINT_HOSTS)


# ============================================================
# OPENALEX FETCHING
# ============================================================

# Gets the OpenAlex records for one query and one publication type (article or proceedings-article).
def fetch_openalex(query, publication_type):
    """
    Retrieve OpenAlex records for one query and one scholarly
    publication type.

    publication_type:
        article
        proceedings-article
    """

    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "filter": (
            f"from_publication_date:{START_YEAR}-01-01,"
            f"to_publication_date:{END_YEAR}-12-31,"
            f"type:{publication_type}"
        ),
        "per-page": PER_PAGE,
        "page": 1,
    }

    response = SESSION.get(
        url,
        params=params,
        timeout=45,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("results", [])


# ============================================================
# PAPER CONVERSION
# ============================================================

# Converts an OpenAlex work into the MongoDB paper format.
# It rejects works with a missing year, a year outside the range, or no usable PDF,
# otherwise it copies title, authors, year, DOI, abstract, venue, links and flags.
def convert(work, retrieval_query):
    """
    Convert an OpenAlex work into the MongoDB paper schema.
    """

    if not isinstance(work, dict):
        return None

    # --------------------------------------------------------
    # Publication year
    # --------------------------------------------------------

    year = work.get("publication_year")

    if not year:
        return None

    try:
        year = int(year)
    except (TypeError, ValueError):
        return None

    if year < START_YEAR or year > END_YEAR:
        return None

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = clean(work.get("title"))

    if not title:
        return None

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_url = get_pdf(work)

    if not pdf_url:
        return None

    # --------------------------------------------------------
    # Preprint filtering
    # --------------------------------------------------------

    if is_preprint(work, pdf_url):
        return None

    # --------------------------------------------------------
    # Authors
    # --------------------------------------------------------

    authors = []

    for author in work.get("authorships") or []:

        if not isinstance(author, dict):
            continue

        author_data = author.get("author") or {}

        name = clean(author_data.get("display_name"))

        if name:
            authors.append(name)

    # --------------------------------------------------------
    # Primary source
    # --------------------------------------------------------

    primary = work.get("primary_location") or {}

    if not isinstance(primary, dict):
        primary = {}

    source = primary.get("source") or {}

    if not isinstance(source, dict):
        source = {}

    venue = clean(source.get("display_name"))

    landing = clean(primary.get("landing_page_url"))

    # --------------------------------------------------------
    # DOI
    # --------------------------------------------------------

    doi = normalize_doi(work.get("doi"))

    # --------------------------------------------------------
    # OpenAlex ID
    # --------------------------------------------------------

    openalex_id = clean(work.get("id"))

    # --------------------------------------------------------
    # Paper ID
    # --------------------------------------------------------

    paper_id = openalex_id or doi or normalize_text(
        f"{title} {year}"
    )

    if not paper_id:
        return None

    # --------------------------------------------------------
    # Abstract
    # --------------------------------------------------------

    abstract = get_abstract(work)

    # --------------------------------------------------------
    # Publication type
    # --------------------------------------------------------

    publication_type = get_publication_type(work)

    # --------------------------------------------------------
    # Return MongoDB-compatible record
    # --------------------------------------------------------

    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "publication_date": work.get("publication_date"),
        "abstract": abstract,
        "doi": doi,
        "venue": venue,
        "publication_types": [publication_type],
        "url": landing,
        "pdf_url": pdf_url,
        "source": "OpenAlex",
        "is_open_access": True,
        "is_preprint": False,
        "is_available": True,
        "methodology": "",
        "results": "",
        "evaluation_metrics": [],
        "conclusion": "",
        "limitations": "",
        "retrieval_queries": [retrieval_query],
    }


# ============================================================
# DUPLICATE DETECTION
# ============================================================

# Makes several keys for a paper (by id, by DOI, by title+year) so the same paper found by different
# queries is recognised and added only once.
def make_dedup_keys(paper):
    """
    Generate several keys so duplicate papers found by different
    OpenAlex search queries are removed before MongoDB insertion.
    """

    keys = set()

    paper_id = clean(paper.get("paper_id"))

    if paper_id:
        keys.add(f"id:{paper_id.lower()}")

    doi = normalize_doi(paper.get("doi"))

    if doi:
        keys.add(f"doi:{doi}")

    title = normalize_text(paper.get("title"))
    year = paper.get("year")

    if title:
        keys.add(f"title:{title}:{year}")

    return keys


# ============================================================
# CORPUS COLLECTION
# ============================================================

# Builds the broad corpus: runs many search queries across AI/ML research areas, for article and
# proceedings types, converts the results, removes duplicates with make_dedup_keys, and stops at the target size.
# Existing MongoDB records are not deleted.
def collect_papers():
    """
    Search multiple scholarly concepts and build one deduplicated
    corpus.

    Existing MongoDB records are not deleted. The Node backend
    upserts/inserts the collected records.
    """

    papers = []

    seen_keys = set()

    total_received = 0
    total_eligible = 0
    total_duplicates = 0

    publication_types = [
        "article",
        "proceedings-article",
    ]

    print()
    print("=" * 78)
    print("OPENALEX BROAD SCHOLARLY CORPUS COLLECTION")
    print("=" * 78)
    print(f"Year range : {START_YEAR}-{END_YEAR}")
    print(f"Target     : {TARGET}")
    print(f"Queries    : {len(SEARCH_QUERIES)}")
    print()

    for query_index, query in enumerate(SEARCH_QUERIES, start=1):

        if len(papers) >= TARGET:
            break

        print(
            f"[{query_index:02d}/{len(SEARCH_QUERIES):02d}] "
            f"Query: {query}"
        )

        for publication_type in publication_types:

            if len(papers) >= TARGET:
                break

            print(
                f"    Type: {publication_type}"
            )

            try:
                works = fetch_openalex(
                    query,
                    publication_type,
                )

            except Exception as exc:

                print(
                    f"    OpenAlex request failed: {exc}"
                )

                # Continue with the next query rather than destroying
                # an otherwise useful corpus collection.
                continue

            total_received += len(works)

            print(
                f"    OpenAlex records: {len(works)}"
            )

            added_for_request = 0

            for work in works:

                paper = convert(
                    work,
                    query,
                )

                if not paper:
                    continue

                candidate_keys = make_dedup_keys(paper)

                if not candidate_keys:
                    continue

                # ------------------------------------------------
                # Duplicate check
                # ------------------------------------------------

                if candidate_keys.intersection(seen_keys):

                    total_duplicates += 1
                    continue

                # ------------------------------------------------
                # Accept paper
                # ------------------------------------------------

                seen_keys.update(candidate_keys)

                papers.append(paper)

                total_eligible += 1
                added_for_request += 1

                if len(papers) >= TARGET:
                    break

            print(
                f"    Added: {added_for_request}"
            )

            time.sleep(REQUEST_DELAY_SECONDS)

    print()
    print("-" * 78)
    print("COLLECTION SUMMARY")
    print("-" * 78)
    print(f"OpenAlex records received : {total_received}")
    print(f"Eligible unique papers    : {total_eligible}")
    print(f"Duplicates removed        : {total_duplicates}")
    print(f"Final corpus batch         : {len(papers)}")
    print()

    return papers


# ============================================================
# NODE / MONGODB IMPORT
# ============================================================

# Sends the collected papers to the Node server (POST /api/corpus/papers), which inserts or updates them in MongoDB.
# Prints the server's answer.
def import_to_node(papers):
    """
    Send the collected papers to the existing Node.js backend.

    The Node endpoint is responsible for inserting/upserting them
    into MongoDB.
    """

    if not papers:
        print(
            "No eligible papers with accessible scholarly PDF URLs "
            "were found."
        )
        return False

    print("=" * 78)
    print("IMPORTING PAPERS INTO NODE / MONGODB")
    print("=" * 78)
    print(f"Endpoint: {NODE_URL}")
    print(f"Papers  : {len(papers)}")
    print()

    try:

        response = SESSION.post(
            NODE_URL,
            json={
                "papers": papers,
            },
            timeout=180,
        )

        print(
            "Node response status:",
            response.status_code,
        )

        try:
            print(
                "Node response:",
                response.json(),
            )
        except Exception:
            print(
                "Node response:",
                response.text,
            )

        response.raise_for_status()

    except Exception as exc:

        print()
        print(
            "MongoDB corpus import failed:",
            exc,
        )

        return False

    return True


# ============================================================
# MAIN
# ============================================================

# Runs the corpus-filling script: prints the plan, calls collect_papers(), then import_to_node().
def main():

    print()
    print("=" * 78)
    print("MULTI-AGENT LITERATURE REVIEW GENERATOR")
    print("SCHOLARLY CORPUS POPULATION")
    print("=" * 78)
    print()
    print(
        "This script builds a broad reusable scholarly corpus."
    )
    print(
        "It is NOT restricted to one application domain."
    )
    print()
    print(
        f"Years  : {START_YEAR}-{END_YEAR}"
    )
    print(
        f"Target : {TARGET} unique papers"
    )
    print()

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    try:

        papers = collect_papers()

    except KeyboardInterrupt:

        print()
        print("Collection interrupted by user.")
        sys.exit(1)

    except Exception as exc:

        print()
        print(
            "Corpus collection failed:",
            exc,
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not papers:

        print()
        print(
            "No eligible papers were collected."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    print("=" * 78)
    print("SAMPLE COLLECTED PAPERS")
    print("=" * 78)

    for index, paper in enumerate(
        papers[:10],
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"{paper['title']} "
            f"({paper['year']})"
        )

    if len(papers) > 10:
        print(
            f"... and {len(papers) - 10} more"
        )

    print()

    # --------------------------------------------------------
    # Import
    # --------------------------------------------------------

    success = import_to_node(papers)

    if not success:
        sys.exit(1)

    print()
    print("=" * 78)
    print("CORPUS IMPORT COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print()
    print(
        "The MongoDB corpus has now been expanded with "
        "multi-domain scholarly literature."
    )
    print()


if __name__ == "__main__":
    main()