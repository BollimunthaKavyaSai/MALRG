"""
Stages 1-4 of the pipeline (Review-1 slide 22):

  1. Query Understanding Agent  - decomposes / expands the topic (LLM)
  2. Scholarly Retrieval        - live OpenAlex (peer-reviewed, open access) + arXiv
  3. Hybrid Ranking             - 60% semantic (MiniLM) + 20% TF-IDF + 15% title + 5% quality
  4. PDF Extraction Agent       - PyMuPDF full text, section detection, passage index

The selected papers for a topic are frozen to data/benchmark/<slug>.json so that
every ablation configuration is evaluated on exactly the same source set.
"""

import hashlib
import json
import math
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests

import config
from malrg import llm
from malrg.text_utils import chunk_text, normalize_title, normalize_ws

UA = {"User-Agent": "MALRG-research-evaluation/1.0 (academic project; mailto:malrg-eval@example.org)"}
_EMBEDDER = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    return _EMBEDDER


def slugify(topic: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return s[:60]


# ---------------------------------------------------------------------------
# 1. Query understanding
# ---------------------------------------------------------------------------
def understand_query(topic: str):
    prompt = f"""You are the Query Understanding Agent of a scholarly literature-review system.
Research topic: "{topic}"

Return JSON with:
- "research_intent": one sentence describing what a literature review on this topic must cover
- "core_concepts": 2-4 concepts that a relevant paper MUST be about
- "subtopics": 3-5 important sub-areas
- "search_queries": 4 short keyword queries (3-7 words each) for an academic search engine,
  covering the core topic and its main subtopics"""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="query", max_tokens=600, default={})
    queries = [topic] + [q for q in out.get("search_queries", []) if isinstance(q, str)]
    seen, uniq = set(), []
    for q in queries:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(q.strip())
    out["search_queries"] = uniq[:5]
    out.setdefault("research_intent", topic)
    out.setdefault("core_concepts", [])
    return out


# ---------------------------------------------------------------------------
# 2. Live scholarly retrieval
# ---------------------------------------------------------------------------
def _openalex_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def search_openalex(query, n):
    params = {
        "search": query,
        "filter": f"is_oa:true,type:article|review,from_publication_date:{config.START_YEAR}-01-01,"
                  f"to_publication_date:{config.END_YEAR}-12-31",
        "per-page": n,
        "select": "id,doi,title,publication_year,authorships,primary_location,best_oa_location,"
                  "cited_by_count,abstract_inverted_index,type",
    }
    for attempt in range(3):
        try:
            r = requests.get("https://api.openalex.org/works", params=params, headers=UA, timeout=40)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    else:
        return []
    out = []
    for w in r.json().get("results", []):
        title = normalize_ws(w.get("title"))
        abstract = _openalex_abstract(w.get("abstract_inverted_index"))
        if not title or len(abstract.split()) < 40:
            continue
        loc = w.get("best_oa_location") or {}
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        out.append({
            "id": w["id"].split("/")[-1],
            "title": title,
            "abstract": abstract,
            "year": w.get("publication_year"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:6]
                        if a.get("author")],
            "venue": venue,
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "url": w.get("doi") or w["id"],
            "pdf_url": loc.get("pdf_url"),
            "citations": w.get("cited_by_count") or 0,
            "source": "OpenAlex",
            "query": query,
        })
    return out


def search_arxiv(query, n):
    terms = " AND ".join(f'all:"{t}"' if " " in t else f"all:{t}"
                         for t in re.findall(r"[A-Za-z0-9\-]+", query)[:6])
    params = {"search_query": terms, "max_results": n, "sortBy": "relevance"}
    for attempt in range(3):
        try:
            r = requests.get("http://export.arxiv.org/api/query", params=params, headers=UA, timeout=40)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(4 * (attempt + 1))
    else:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []
    for e in root.findall("a:entry", ns):
        title = normalize_ws(e.findtext("a:title", "", ns))
        abstract = normalize_ws(e.findtext("a:summary", "", ns))
        year = int((e.findtext("a:published", "", ns) or "0")[:4] or 0)
        if not title or year < config.START_YEAR or year > config.END_YEAR:
            continue
        arxiv_id = e.findtext("a:id", "", ns).split("/abs/")[-1]
        out.append({
            "id": "arXiv:" + arxiv_id,
            "title": title,
            "abstract": abstract,
            "year": year,
            "authors": [normalize_ws(a.findtext("a:name", "", ns)) for a in e.findall("a:author", ns)][:6],
            "venue": "arXiv",
            "doi": "",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "citations": 0,
            "source": "arXiv",
            "query": query,
        })
    time.sleep(3)  # arXiv API etiquette
    return out


def collect_candidates(topic, intent):
    pool = {}
    for q in intent["search_queries"]:
        for paper in search_openalex(q, config.CANDIDATES_PER_QUERY) + search_arxiv(q, 15):
            key = normalize_title(paper["title"])
            if key and key not in pool:
                pool[key] = paper
    return list(pool.values())


# ---------------------------------------------------------------------------
# 3. Hybrid ranking (and single-signal variants for the retrieval ablation)
# ---------------------------------------------------------------------------
def score_candidates(topic, intent, candidates):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    query_text = f"{topic}. {intent.get('research_intent', '')} " + " ".join(intent.get("core_concepts", []))
    docs = [f"{c['title']}. {c['abstract']}" for c in candidates]

    emb = embedder()
    qv = emb.encode([query_text], normalize_embeddings=True)
    dv = emb.encode(docs, normalize_embeddings=True, batch_size=32)
    semantic = (dv @ qv.T).ravel()

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(docs + [query_text])
    tfidf = cosine_similarity(mat[:-1], mat[-1]).ravel()

    concept_tokens = set()
    for c in [topic] + list(intent.get("core_concepts", [])):
        concept_tokens.update(t for t in re.findall(r"[a-z0-9]+", c.lower()) if len(t) > 2)
    title_scores = []
    for c in candidates:
        tt = set(re.findall(r"[a-z0-9]+", c["title"].lower()))
        title_scores.append(len(tt & concept_tokens) / max(1, len(concept_tokens)))
    title_scores = np.array(title_scores)

    cites = np.array([math.log1p(c.get("citations") or 0) for c in candidates])
    recency = np.array([max(0, (c.get("year") or 2017) - 2016) / 10 for c in candidates])
    quality = 0.6 * _minmax(cites) + 0.2 * recency + 0.2 * np.array(
        [1.0 if c["source"] == "OpenAlex" else 0.5 for c in candidates])

    sig = {"semantic": _minmax(semantic), "tfidf": _minmax(tfidf),
           "title": _minmax(title_scores), "quality": _minmax(quality)}
    w = config.RANK_WEIGHTS
    hybrid = sum(w[k] * sig[k] for k in w)
    for i, c in enumerate(candidates):
        c["scores"] = {k: round(float(sig[k][i]), 4) for k in sig}
        c["scores"]["hybrid"] = round(float(hybrid[i]), 4)
    return candidates


def _minmax(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / np.ptp(x)


SURVEY_RE = re.compile(r"\b(review|survey|overview|state of the art|state-of-the-art)\b", re.I)

RANKERS = ["hybrid", "semantic", "tfidf", "title"]


def ranking(candidates, ranker):
    return sorted(candidates, key=lambda c: c["scores"][ranker], reverse=True)


# ---------------------------------------------------------------------------
# 4. PDF extraction
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "abstract": r"abstract",
    "introduction": r"introduction|background",
    "method": r"method|methodology|approach|proposed|framework|model architecture|materials and methods|system design",
    "results": r"results?|experiments?|evaluation|performance|findings|case study",
    "discussion": r"discussion|analysis",
    "limitations": r"limitations?|threats to validity|challenges|open issues|future (work|directions?)",
    "conclusion": r"conclusions?|concluding remarks|summary",
}


# hosts that serve an HTML/JS challenge instead of the PDF to scripted clients
BLOCKING_HOSTS = ("ieeexplore.ieee.org", "sciencedirect.com", "onlinelibrary.wiley.com",
                  "researchgate.net", "tandfonline.com", "academic.oup.com")


def download_pdf(paper):
    url = paper.get("pdf_url")
    if not url:
        return None
    fname = hashlib.md5(url.encode()).hexdigest() + ".pdf"
    path = config.PDF_DIR / fname
    if path.exists() and path.stat().st_size > 10_000:
        return path
    if any(h in url for h in BLOCKING_HOSTS):
        return None
    try:
        r = requests.get(url, headers={**UA, "Accept": "application/pdf"}, timeout=(8, 25),
                         allow_redirects=True)
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.content.startswith(b"%PDF") or len(r.content) < 10_000:
        return None
    path.write_bytes(r.content)
    return path


def extract_fulltext(pdf_path, max_pages=30):
    import fitz
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pages.append(page.get_text("text"))
    doc.close()
    text = "\n".join(pages)
    # cut the bibliography: it is not evidence and pollutes the passage index
    m = list(re.finditer(r"\n\s*(references|bibliography|literature cited)\s*\n", text, re.I))
    if m and m[-1].start() > len(text) * 0.5:
        text = text[:m[-1].start()]
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text


def detect_sections(text):
    """Map canonical section -> text using heading heuristics on the raw PDF text."""
    lines = text.split("\n")
    heads = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) > 70:
            continue
        s2 = re.sub(r"^((\d+(\.\d+)*)|[IVX]+)[.)]?\s+", "", s).strip().lower()
        for name, pat in SECTION_PATTERNS.items():
            if re.fullmatch(rf"({pat})( and \w+)?( \w+)?", s2):
                heads.append((idx, name))
                break
    sections = {}
    for j, (idx, name) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
        body = normalize_ws(" ".join(lines[idx + 1:end]))
        if len(body.split()) > 30:
            sections[name] = (sections.get(name, "") + " " + body).strip()
    return sections


def build_paper_record(paper):
    pdf = download_pdf(paper)
    if not pdf:
        return None
    raw = extract_fulltext(pdf)
    full = normalize_ws(raw)
    if len(full.split()) < config.MIN_FULLTEXT_WORDS:
        return None
    sections = detect_sections(raw)
    sections.setdefault("abstract", paper["abstract"])
    passages = chunk_text(full, config.CHUNK_WORDS, config.CHUNK_OVERLAP)
    rec = dict(paper)
    rec.update({"fulltext_words": len(full.split()), "sections": sections, "passages": passages,
                "pdf_file": pdf.name})
    return rec


# ---------------------------------------------------------------------------
# Benchmark construction (frozen per topic)
# ---------------------------------------------------------------------------
def benchmark_path(topic):
    return config.BENCHMARK_DIR / f"{slugify(topic)}.json"


def build_benchmark(topic, log=print):
    path = benchmark_path(topic)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    log(f"  [retrieval] query understanding: {topic}")
    intent = understand_query(topic)
    log(f"  [retrieval] queries: {intent['search_queries']}")
    candidates = collect_candidates(topic, intent)
    log(f"  [retrieval] {len(candidates)} unique candidates (OpenAlex + arXiv)")
    candidates = score_candidates(topic, intent, candidates)

    selected = []
    n_surveys = 0
    for cand in ranking(candidates, "hybrid"):
        if len(selected) >= config.PAPERS_PER_TOPIC:
            break
        is_survey = bool(SURVEY_RE.search(cand["title"]))
        if is_survey and n_surveys >= config.MAX_SURVEYS_PER_TOPIC:
            continue  # prefer primary studies that report methods and results
        rec = build_paper_record(cand)
        if rec is None:
            continue  # publisher PDF blocked or too short: take the next ranked paper
        selected.append(rec)
        n_surveys += is_survey
        log(f"  [extraction] + {rec['title'][:80]} ({rec['fulltext_words']} words)")

    for i, p in enumerate(selected, 1):
        p["ref_id"] = i

    bench = {
        "topic": topic,
        "slug": slugify(topic),
        "intent": intent,
        "candidates": [{k: v for k, v in c.items() if k not in ("passages", "sections")}
                       for c in candidates],
        "papers": selected,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(bench, ensure_ascii=False, indent=1), encoding="utf-8")
    return bench
