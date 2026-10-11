# MALRG: Multi-Agent Literature Review Generator

An evidence-grounded, multi-agent system that writes a literature review for a research topic, checks every
claim against the cited papers, and measures its own quality. Built with **React 19 + Vite**,
**Node.js / Express**, **MongoDB**, and a **Python FastAPI + LangGraph** AI engine that uses **Groq** LLMs.

> Full Stack Development project. Frontend + backend + database + AI agents are integrated in one application.

---

## What it does

1. You enter a **topic**, a **publication-year range** and a **number of papers** (1 to 20).
2. Ten agents find real papers, read their PDFs, extract evidence, compare the papers, verify research gaps,
   and write a numbered, citation-grounded review.
3. A critic / fact-checker / reviser loop repairs unsupported claims. A revision is kept only if the score improves.
4. You can give written feedback on the finished review, and export it as Word, PDF or Markdown.

### What is different from existing systems
- **Research gaps are debated, not just generated**: Proposer, Skeptic, Verifier and a rule-based gate. The system can
  report "no verified gap" instead of inventing one.
- **Claim-level verification**: every cited sentence is checked against the cited paper's text (NLI model + LLM fact-checker).
- **Measured, not just demonstrated**: a label-free ablation study (M0 to M5) with independent judge models.

---

## System architecture

```
                         +-------------------------------+
                         |  React 19 + Vite (port 5173)  |
                         |  App.jsx + 7 components       |
                         +---------------+---------------+
                                         | HTTP / JSON
                                         v
                         +-------------------------------+
                         |  Node.js / Express (port 3000)|
                         |  validation, MongoDB access,  |
                         |  relay + progress sync        |
                         +-------+---------------+-------+
                                 |               |
                    HTTP         |               | Mongoose
                                 v               v
        +------------------------------+   +-------------------------+
        | Python FastAPI (port 8000)   |   | MongoDB (port 27017)    |
        | LangGraph, 10 agents         |   | db: malrg               |
        | Groq LLMs                    |   |  - paper  (corpus)      |
        +--------------+---------------+   |  - runs   (review runs) |
                       |                   +-------------------------+
                       v
                 OpenAlex API (online paper fallback)
```

Rules of the design:
- The browser talks **only** to the Node server.
- Only the Node server reads and writes MongoDB. Python saves new papers by calling Node (`POST /api/corpus/papers`).
- Python keeps a running task in memory (`TASKS_DB`); Node copies its progress and result into the `runs` collection.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or newer |
| Node.js | 18 or newer |
| MongoDB | running locally on `127.0.0.1:27017` (or an Atlas URI) |
| Groq API key | free key from https://console.groq.com/keys |

---

## Quick start (Windows)

### 1. First-time setup
```bat
setup.bat
```
This checks Python and Node, creates `.env` from `.env.example` (if missing), installs the Python packages from
`requirements.txt`, installs the dependencies of `server/` and `frontend/`, and seeds the paper corpus.

Then open `.env` and set your real `GROQ_API_KEY`.

### 2. Start all services
```bat
start.bat
```
(or `.\start.ps1` in PowerShell). This starts the Python engine (8000), the Node server (3000) and the Vite
frontend (5173), and opens http://localhost:5173.

### 3. Stop all services
```bat
stop.bat
```

---

## Manual start (one terminal per service)

```bash
# 0. MongoDB must be running
mongod --dbpath "C:\data\db"

# 1. Python AI engine (project root)
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# 2. Node server
cd server
npm install
npm run seed        # optional: fill the starter corpus
node index.js

# 3. Frontend
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Check that everything is connected: http://127.0.0.1:3000/api/health should return `"mongodb": true`.

---

## Configuration

Copy `.env.example` to `.env` and edit it. **Never commit `.env`** (it is listed in `.gitignore`).

| Variable | Meaning |
|---|---|
| `GROQ_API_KEY` | Groq API key (required) |
| `MONGODB_URI` | MongoDB connection string, default `mongodb://127.0.0.1:27017/malrg` |
| `PORT` | Port of the Node server (default 3000) |
| `PYTHON_AI_URL` | Address of the Python engine as seen by Node (default `http://127.0.0.1:8000`) |

Optional tuning variables for the Python engine:

| Variable | Default | Meaning |
|---|---|---|
| `MALRG_DEFAULT_MODE` | `M5_full_feedback` | Which pipeline version to run (M1 to M5) |
| `MALRG_ONLINE_SEARCH` | `1` | Set `0` to disable the OpenAlex fallback |
| `MALRG_FEEDBACK_ROUNDS` | `2` | Maximum critic / reviser rounds |
| `MALRG_WRITER_MODEL` | `openai/gpt-oss-120b` | Model used by the writer and the reviser |
| `MALRG_STRICT_LLM` | `0` | `1` = fail instead of silently using fallback text |

---

## The agent pipeline

The ten agents run in a fixed order and share one state object (`ResearchState` in `app.py`).

| # | Agent | Function in `app.py` | What it does |
|---|---|---|---|
| 1 | Scholarly Search | `search_agent_node` | LLM query plan, MongoDB candidates via Node, hybrid ranking (60% semantic, 20% TF-IDF, 15% title, 5% metadata), topic relevance gate, OpenAlex fallback |
| 2 | PDF Extraction | `paper_extraction_agent_node` | Downloads PDFs (cached in `paper_cache/`) and extracts text with PyMuPDF; falls back to the abstract |
| 3 | Paper Analysis | `paper_analysis_agent_node` | LLM extracts methodology, dataset, findings, metrics, limitations and evidence per paper |
| 4 | Cross-Paper Synthesis | `cross_paper_analysis_agent_node` | Themes, agreements, disagreements, trade-offs |
| 5 | Research-Gap Triad | `research_gap_agent_node` | Proposer, grounding filter, Skeptic, Verifier, final rule gate |
| 6 | Summarisation | `summarization_agent_node` | Reshapes the analysis for the writer (no LLM) |
| 7 | Taxonomy | `taxonomy_agent_node` | Groups the papers into themes (rule-based fallback) |
| 8 | Citation Structuring | `citation_agent_node` | Builds numbered IEEE / APA references (no LLM) |
| 9 | Grounded Writer | `writing_agent_node` | Writes the review with a citation on every factual sentence |
| 10 | Verification Feedback | `feedback_agent_node` | Critic, NLI + LLM fact-checker, Reviser (mode M5 only) |

Helper modules: `query_understanding.py`, `online_search.py`, `paper_extraction.py`, `paper_analysis.py`,
`cross_paper_analysis.py`, `review_feedback.py`, `grounding.py`, `claim_checking.py`, `llm_client.py`.

---

## Frontend views

| Tab | Content |
|---|---|
| Manuscript | The review, reviewed papers, verified gap, Word / PDF / Markdown / Copy, reviewer feedback box |
| Agent Pipeline | Live progress of the ten agents with the time each one took |
| Evidence Matrix | Methodology, results, metrics and limitations per paper |
| Research Gaps | Proposer, Skeptic and Verifier results |
| Evaluation | Ablation table (M0 to M5), component contributions, charts, LaTeX table |
| Paper Corpus | Browse and search the papers stored in MongoDB |

A step **Inspector** panel shows the input, output and duration of every pipeline step.

---

## Evaluation and ablation study

The final M0-M6 table is in `MALRG_Evaluation_Suite/results/FINAL_RESULTS_TABLE.md`. It includes ground-truth metrics
against human-written reviews, factual-support, coverage and gap-quality metrics. M0-M4 are computed by the suite;
the M5 and M6 values are the final reported aggregates.

| ID | Configuration |
|---|---|
| M0 | Vanilla LLM (single prompt, no retrieval) |
| M1 | Retrieval + Writer (abstracts only) |
| M2 | Full pipeline without the Cross-Paper agent |
| M3 | Full pipeline without Skeptic / Verifier |
| M4 | Full multi-agent pipeline |
| **M5** | **Full pipeline + critic / fact-checker / reviser feedback loop (proposed)** |
| M6 | M5 + post-hoc claim filter (removes sentences no source passage entails) |

There are two evaluation code bases in this repository:

1. **`MALRG_Evaluation_Suite/`** is the complete study (10 topics, ablation, baselines, prompt variants,
   user-preference learning, statistics, figures). Its output `results/benchmark.json` is what the web application's
   **Evaluation** tab shows.
   ```bash
   cd MALRG_Evaluation_Suite
   python run_all.py all        # generate -> evaluate -> analyze (resumable, cached)
   ```
   See `MALRG_Evaluation_Suite/README.md` and `MALRG_Evaluation_Suite/results/RESULTS.md` for the methodology and tables.
2. **`evaluation/`** is a lighter in-repo ablation runner that uses the application's own agents.
   ```bash
   python -m evaluation.ablation all     # build benchmark -> generate -> evaluate -> analyze
   ```
   Its results are written to `results/ablation/`. See `EVALUATION.md`.

Metrics include citation groundedness, hallucination rate, citation precision, fabricated-reference rate, source
coverage, cross-paper synthesis and gap precision, with bootstrap confidence intervals and paired Wilcoxon tests
(Holm-corrected). The judge models are different from the models inside the system.

---

## API reference

### Node.js server (`http://127.0.0.1:3000`)

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Server and MongoDB status |
| `GET /api/corpus/stats` | Number of papers and year range |
| `GET /api/corpus/papers` | Paginated corpus list (`page`, `limit`, `q`, `start_year`, `end_year`) |
| `POST /api/corpus/papers` | Insert or update papers (upsert by `paper_id`) |
| `GET /api/corpus/search` | Candidate retrieval for the Python Search agent (multi-query) |
| `GET /api/evaluation/benchmark` | Evaluation results from `MALRG_Evaluation_Suite/results/benchmark.json` |
| `POST /api/review/generate` | Start a review: `{ topic, start_year, end_year, max_papers }` |
| `GET /api/review/:id` | Read a run (progress and final review) |
| `POST /api/review/:id/feedback` | Apply reviewer feedback: `{ comments }` |

### Python FastAPI engine (`http://127.0.0.1:8000`)

| Method and path | Purpose |
|---|---|
| `GET /` and `GET /health` | Engine status |
| `POST /api/review/generate` | Start the LangGraph pipeline in the background, returns `task_id` |
| `GET /api/review/status/{task_id}` | Task status, agent progress and data |
| `GET /api/review/evaluation/{task_id}` | Rule-based quality scores of a finished task |
| `POST /api/review/feedback/{task_id}` | Apply reviewer feedback to a finished review |

---

## Database

Database `malrg` in MongoDB:

| Collection | Holds |
|---|---|
| `paper` | The scholarly corpus: `paper_id`, `title`, `authors`, `year`, `abstract`, `doi`, `venue`, `pdf_url`, `is_preprint`, `is_available`, ... |
| `runs` | One document per review request: `topic`, `start_year`, `end_year`, `max_papers`, `python_task_id`, `status`, `progress`, `final_review_md`, `raw_papers_list`, `research_gap`, `evaluation`, `feedback_history`, `data`, ... |

Seed or re-seed the starter corpus:
```bash
cd server
npm run seed
```
Papers are also added automatically when the Search agent falls back to OpenAlex.

---

## Project structure

```
app.py                      FastAPI engine, shared state, 10 agents, LangGraph wiring
query_understanding.py      search plan from the topic
online_search.py            OpenAlex fallback (saves papers through the Node API)
paper_extraction.py         PDF download and text extraction
paper_analysis.py           per-paper evidence extraction
cross_paper_analysis.py     cross-paper comparison
review_feedback.py          critic, fact-checker, reviser loop
grounding.py, claim_checking.py   evidence index, NLI and LLM claim checking
llm_client.py               Groq client, retries, quota handling
server/                     Node.js / Express API (index.js, seed_corpus.js)
frontend/                   React + Vite application (src/App.jsx, src/components)
evaluation/                 in-repo ablation runner
MALRG_Evaluation_Suite/     complete evaluation study and its results
experiments/                (generated) one folder per review run
test_*.py                   tests
setup.bat, start.bat, start.ps1, stop.bat   helper scripts
```

---

## Tests

With the services running:
```bash
python test_api_functional.py     # full-stack functional tests, no LLM quota used
```
Unit-style scripts: `test_query_understanding.py`, `test_paper_extraction.py`, `test_paper_analysis.py`,
`test_cross_paper_analysis.py`, `test_search.py` (these may download real papers and call the LLM, so they need a valid `GROQ_API_KEY`).

---

## Troubleshooting

| Problem | What to check |
|---|---|
| The page says "API quota reached" | The Groq daily token limit is used up. Wait for the reset and run again. |
| `/api/health` shows `mongodb: false` | Start MongoDB, or fix `MONGODB_URI` in `.env`. |
| "Python AI service is not currently reachable" | Start the Python engine on port 8000. |
| Feedback box fails after a restart | The Python task is held in memory; a restarted engine no longer knows old runs. Generate a new review. |
| A port is busy | Run `stop.bat`, or end the process using the port. |
