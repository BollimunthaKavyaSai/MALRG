"""
Replace the template seed corpus with the 50 real full-text studies of the evaluation benchmark.

1. Backs up the current `paper` collection to `paper_seed_backup` (nothing is lost; reversible).
2. Clears the active `paper` collection.
3. Inserts the real studies through the application's own REST API (POST /api/corpus/papers),
   including the method, results, metrics and limitations extracted by the Paper Analysis Agent.

    python load_real_corpus.py            (Node API must be running on port 3000)
"""

import json
from pathlib import Path

import requests
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "MALRG_Evaluation_Suite"
API = "http://127.0.0.1:3000/api/corpus/papers"


# Reads the MongoDB address from the MONGODB_URI line of the .env file;
# if it is missing, uses the local default mongodb://127.0.0.1:27017/malrg.
def mongo_uri():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("MONGODB_URI="):
            return line.split("=", 1)[1].strip()
    return "mongodb://127.0.0.1:27017/malrg"


# Builds paper records from the 50 real studies of the evaluation benchmark.
# For each benchmark file it reads the papers and, if available, the Paper Analysis results saved for mode M4
# (method, results, metrics, limitations) and shapes everything like a corpus paper.
def records():
    out = []
    for bench_file in sorted((SUITE / "data" / "benchmark").glob("*.json")):
        bench = json.loads(bench_file.read_text(encoding="utf-8"))
        run_file = SUITE / "runs" / bench["slug"] / "M4.json"
        analyses = {}
        if run_file.exists():
            analyses = {a["ref_id"]: a for a in json.loads(run_file.read_text(encoding="utf-8")).get("analyses") or []}
        for p in bench["papers"]:
            a = analyses.get(p["ref_id"], {})
            join = lambda v: "; ".join(map(str, v)) if isinstance(v, list) else (v or "")
            out.append({
                "paper_id": p["id"],
                "title": p["title"],
                "authors": p.get("authors") or [],
                "year": p.get("year"),
                "abstract": p.get("abstract"),
                "doi": p.get("doi") or "",
                "venue": p.get("venue") or "",
                "url": p.get("url"),
                "pdf_url": p.get("pdf_url"),
                "source": "arXiv" if p["source"] == "arXiv" else "OpenAlex/Peer-Reviewed",
                "is_open_access": True,
                "is_preprint": p["source"] == "arXiv",
                "is_available": True,
                "methodology": join(a.get("method")),
                "results": join(a.get("key_results")),
                "evaluation_metrics": [str(m) for m in (a.get("metrics") or [])],
                "conclusion": join(a.get("contributions")),
                "limitations": join(a.get("limitations")),
                "retrieval_queries": [bench["topic"]],
            })
    return out


# Replaces the sample corpus with the real studies:
#  1. backs up the current 'paper' collection to 'paper_seed_backup' (nothing is lost),
#  2. clears the active collection,
#  3. inserts the real studies through the Node API (POST /api/corpus/papers).
# The Node server must be running on port 3000.
def main():
    db = MongoClient(mongo_uri(), serverSelectionTimeoutMS=3000).get_default_database()
    active = db["paper"]
    seed = list(active.find({}))
    if seed and "paper_seed_backup" not in db.list_collection_names():
        db["paper_seed_backup"].insert_many(seed)
        print(f"backed up {len(seed)} template papers to 'paper_seed_backup'")
    active.delete_many({})
    recs = records()
    r = requests.post(API, json=recs, timeout=60)
    r.raise_for_status()
    print("API response:", r.json())
    print("active corpus size:", active.count_documents({}))


if __name__ == "__main__":
    main()
