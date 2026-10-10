"""
Functional tests of the running MALRG stack (React -> Express -> MongoDB / FastAPI).
Start the services first (start.ps1), then run:

    python test_api_functional.py

Only read-only or validation requests are sent, so the tests do not use any LLM quota.
Results are written to results/functional_tests.json.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API = os.getenv("MALRG_API_URL", "http://127.0.0.1:3000/api")
AI = os.getenv("PYTHON_AI_URL", "http://127.0.0.1:8000")
WEB = os.getenv("MALRG_WEB_URL", "http://127.0.0.1:5173")


# Sends one HTTP request (GET/POST), measures the time and returns (status code, JSON answer, milliseconds).
def call(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    ms = (time.perf_counter() - start) * 1000
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = raw.decode(errors="ignore")
    return status, payload, ms


# Returns the id of a finished run for tests that need one; it is read from the MALRG_TEST_RUN environment variable.
def latest_completed_run_id():
    """The newest completed run id is passed in by the caller through MALRG_TEST_RUN (optional)."""
    return os.getenv("MALRG_TEST_RUN", "")


CASES = []


# Decorator that registers a test: stores its id, title, expected result and function in the CASES list.
def case(cid, title, expected):
    # Inner part of the case decorator: adds the test function to CASES and returns it unchanged.
    def wrap(fn):
        CASES.append((cid, title, expected, fn))
        return fn
    return wrap


# Test case TC01: Express health check reports MongoDB connected.
# Expected result: 200, mongodb = true.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC01", "Express health check reports MongoDB connected", "200, mongodb = true")
def _():
    s, p, ms = call("GET", f"{API}/health")
    return s == 200 and p.get("mongodb") is True, f"{s}, mongodb = {p.get('mongodb')}", ms


# Test case TC02: AI engine health check.
# Expected result: 200, status = ok.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC02", "AI engine health check", "200, status = ok")
def _():
    s, p, ms = call("GET", f"{AI}/health")
    return s == 200 and p.get("status") == "ok", f"{s}, status = {p.get('status')}", ms


# Test case TC03: Frontend is served by Vite.
# Expected result: 200, HTML page.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC03", "Frontend is served by Vite", "200, HTML page")
def _():
    start = time.perf_counter()
    with urllib.request.urlopen(WEB, timeout=30) as r:
        html = r.read().decode(errors="ignore")
        ok = r.status == 200 and "<div id=\"root\">" in html
        return ok, f"{r.status}, root element present = {ok}", (time.perf_counter() - start) * 1000


# Test case TC04: Corpus statistics from MongoDB.
# Expected result: 200, total > 0, year range.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC04", "Corpus statistics from MongoDB", "200, total > 0, year range")
def _():
    s, p, ms = call("GET", f"{API}/corpus/stats")
    yr = p.get("year_range", {})
    return s == 200 and p.get("total", 0) > 0, f"{s}, total = {p.get('total')}, eligible = {p.get('eligible_published')}, years {yr.get('min')}–{yr.get('max')}", ms


# Test case TC05: Paginated corpus listing.
# Expected result: 200, at most 5 papers, page metadata.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC05", "Paginated corpus listing", "200, at most 5 papers, page metadata")
def _():
    s, p, ms = call("GET", f"{API}/corpus/papers?page=1&limit=5")
    n = len(p.get("papers", []))
    return s == 200 and 0 < n <= 5 and p.get("totalPages", 0) >= 1, f"{s}, {n} papers, {p.get('totalPages')} pages", ms


# Test case TC06: Keyword search in the corpus.
# Expected result: 200, every result matches 'plant'.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC06", "Keyword search in the corpus", "200, every result matches 'plant'")
def _():
    s, p, ms = call("GET", f"{API}/corpus/papers?q=plant&limit=20")
    papers = p.get("papers", [])
    ok = s == 200 and papers and all("plant" in json.dumps(x).lower() for x in papers)
    return bool(ok), f"{s}, {len(papers)} matching papers", ms


# Test case TC07: Agent corpus search with year filter.
# Expected result: 200, all papers within 2020–2026.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC07", "Agent corpus search with year filter", "200, all papers within 2020–2026")
def _():
    s, p, ms = call("GET", f"{API}/corpus/search?topic=plant%20disease%20detection&start_year=2020&end_year=2026&limit=10")
    papers = p.get("papers", [])
    ok = s == 200 and all(2020 <= int(x.get("year") or 0) <= 2026 for x in papers)
    return ok, f"{s}, {len(papers)} papers, years within range = {ok}", ms


# Test case TC08: Reject review request without topic.
# Expected result: 400 with error message.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC08", "Reject review request without topic", "400 with error message")
def _():
    s, p, ms = call("POST", f"{API}/review/generate", {"start_year": 2020, "end_year": 2025, "max_papers": 4})
    return s == 400, f"{s}, {p.get('error')}", ms


# Test case TC09: Reject reversed year range.
# Expected result: 400 with error message.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC09", "Reject reversed year range", "400 with error message")
def _():
    s, p, ms = call("POST", f"{API}/review/generate", {"topic": "test", "start_year": 2026, "end_year": 2020, "max_papers": 4})
    return s == 400, f"{s}, {p.get('error')}", ms


# Test case TC10: Reject paper count above the engine limit (50 > 20).
# Expected result: 400 with error message.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC10", "Reject paper count above the engine limit (50 > 20)", "400 with error message")
def _():
    s, p, ms = call("POST", f"{API}/review/generate", {"topic": "test", "start_year": 2020, "end_year": 2025, "max_papers": 50})
    return s == 400, f"{s}, {p.get('error')}", ms


# Test case TC11: Unknown run id.
# Expected result: 404 Run not found.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC11", "Unknown run id", "404 Run not found")
def _():
    s, p, ms = call("GET", f"{API}/review/not-a-valid-id")
    return s == 404, f"{s}, {p.get('error')}", ms


# Test case TC12: Empty reviewer feedback is rejected.
# Expected result: 400 with error message.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC12", "Empty reviewer feedback is rejected", "400 with error message")
def _():
    s, p, ms = call("POST", f"{API}/review/000000000000000000000000/feedback", {"comments": " "})
    return s == 400, f"{s}, {p.get('error')}", ms


# Test case TC13: Unknown AI task id on the engine.
# Expected result: 404.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC13", "Unknown AI task id on the engine", "404")
def _():
    s, p, ms = call("GET", f"{AI}/api/review/status/unknown-task")
    return s == 404, f"{s}", ms


# Test case TC14: Evaluation results served to the dashboard.
# Expected result: 200, available, 6+ configurations.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC14", "Evaluation results served to the dashboard", "200, available, 6+ configurations")
def _():
    s, p, ms = call("GET", f"{API}/evaluation/benchmark")
    n = len(p.get("configs", []))
    return s == 200 and p.get("available") and n >= 6, f"{s}, {n} configurations, {len(p.get('charts', []))} charts", ms


# Test case TC15: Completed run is stored with papers, gaps, progress and manuscript.
# Expected result: 200, all parts present.
# The function calls the running service (Node, Python or the web page) with the call() helper,
# compares the answer with what is expected, and returns (passed?, what was seen, time in ms).
# The @case decorator registers it so main() can run every test in order.
@case("TC15", "Completed run is stored with papers, gaps, progress and manuscript", "200, all parts present")
def _():
    rid = latest_completed_run_id()
    if not rid:
        return None, "skipped (set MALRG_TEST_RUN)", 0
    s, p, ms = call("GET", f"{API}/review/{rid}")
    gap = p.get("research_gap") or {}
    parts = {
        "status": p.get("status"),
        "papers": len(p.get("raw_papers_list") or []),
        "agents": len(p.get("progress") or []),
        "verified_gaps": len(gap.get("verified_gaps") or []),
        "words": len((p.get("final_review_md") or "").split()),
    }
    ok = s == 200 and parts["status"] == "completed" and parts["papers"] > 0 and parts["agents"] == 10 and parts["words"] > 300
    return ok, f"{s}, " + ", ".join(f"{k} = {v}" for k, v in parts.items()), ms


# Runs every registered test case, prints Pass/Fail with the time, and saves results/functional_tests.json.
# Only read-only or validation requests are used, so no LLM quota is spent.
def main():
    rows = []
    for cid, title, expected, fn in CASES:
        try:
            ok, actual, ms = fn()
        except Exception as exc:  # a failing request is a failed test, not a crash
            ok, actual, ms = False, f"error: {exc}", 0
        verdict = "Skipped" if ok is None else "Pass" if ok else "Fail"
        rows.append({"id": cid, "test": title, "expected": expected, "actual": actual,
                     "result": verdict, "ms": round(ms, 1)})
        print(f"{cid} {verdict:7} {title}  ->  {actual}  ({ms:.0f} ms)")
    out = Path(__file__).resolve().parent / "results" / "functional_tests.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(r["result"] == "Pass" for r in rows)
    print(f"\n{passed}/{len(rows)} passed -> {out}")


if __name__ == "__main__":
    main()
