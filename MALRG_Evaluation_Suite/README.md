# MALRG Evaluation Suite

End-to-end, fully automatic evaluation of the **Multi-Agent Literature Review Generator (MALRG)**:
evaluation metrics, a seven-configuration ablation study (M0-M6), the verification-guided feedback loop,
and the numerical (statistical) analysis. The M0-M6 results and the ground-truth metrics against the human-written reviews are stored in
`results/final_ablation_values.json`.

See **[results/FINAL_RESULTS_TABLE.md](results/FINAL_RESULTS_TABLE.md)** for the final M0-M6 table and
`results/RESULTS.md` for the remaining auto-generated tables.

## What it does

| Review-1 item | Where it is implemented |
|---|---|
| 1. Evaluation metrics | `evaluation/metrics.py` (NLI grounding, hallucination, citation precision, coverage, synthesis, slide-28 indicators), `evaluation/judge.py` (blind LLM judge, gap labels, silver relevance) |
| 2. Ablation study (M0-M4, proposed M5, M5 + claim filter M6) | `config.py` (`ABLATION_CONFIGS`), `malrg/pipeline.py` (one LangGraph per configuration) |
| 3. Adding feedback | `malrg/agents.py` → `critic()` + `refine_review()` (Verification-Guided Iterative Refinement) and `apply_user_feedback()` (human feedback hook) |
| 4. Numerical analysis (ablation) | `evaluation/stats.py` (bootstrap CIs, Wilcoxon + Holm, Cohen's dz, rank-biserial, win rate, Friedman + Kendall's W, component contributions), `evaluation/figures.py` |
| Baseline models | `config.BASELINE_CONFIGS`: vanilla GPT-OSS-120B (M0), GPT-OSS-20B (B1), Qwen3-27B (B2), RAG baseline (M1) vs. M5 |
| 5-prompt comparison | `malrg/prompts.py` (P1 structured, P2 zero-shot, P3 expert persona, P4 plan-then-write, P5 evidence-anchored), analysed in `evaluation/experiments_report.py` |
| Feedback applied next time | `malrg/preferences.py`: feedback → persistent rules → injected into every future review; measured on held-out topics (M4 vs. U1) |

### Teaching the system your preferences

```bash
python run_all.py feedback "Keep reviews under 900 words and always report exact numbers."
python run_all.py preferences                      # show learned rules
python run_all.py review "Graph neural networks for drug discovery"   # uses them automatically
```

## Pipeline (mirrors the Review-1 architecture)

```
topic -> Query Understanding (LLM) -> OpenAlex + arXiv retrieval -> Hybrid ranking (60/20/15/5)
      -> PDF extraction (PyMuPDF, sections, passage index)          [frozen per topic]
      -> Paper Analysis + Evidence Gate (verbatim-quote check)
      -> Taxonomy (embedding clustering) -> Cross-Paper Synthesis
      -> Gap Proposer -> Evidence Gate -> Skeptic (other model family) -> Grounded Verifier
      -> Citation-Grounded Writer -> [Feedback: NLI critic -> re-citation -> reviser -> verified acceptance]
      -> Citation Structuring Agent (IEEE references from verified metadata)
```

## Running

```bash
pip install -r requirements.txt
# GROQ_API_KEY is read from the environment or from ../.env
python run_all.py all              # generate -> evaluate -> analyze (resumable)
python run_all.py generate --topics 4 --configs M4,M5
python run_all.py evaluate
python run_all.py analyze
```

Every LLM call is cached in `cache/llm/`, so re-running reproduces the exact outputs at zero cost.
If a Groq daily quota is reached, work for that model is postponed and the rest continues.
`auto_resume.ps1` re-runs the study every 30 minutes until all 140 runs (10 topics × 14 configs)
are generated and evaluated:

```bash
powershell -ExecutionPolicy Bypass -File auto_resume.ps1
```

Progress: `logs/auto_resume.log`. `results/` is refreshed after every attempt.

## Layout

```
config.py              all experiment parameters (topics, models, thresholds, configs)
run_all.py             CLI driver
malrg/                 the multi-agent system (retrieval, agents, LangGraph pipeline, LLM client)
evaluation/            NLI verifier, metrics, LLM judge, statistics, figures, report builder
data/benchmark/        frozen per-topic source sets (metadata, sections, passages)
data/pdfs/             downloaded open-access PDFs
runs/<topic>/          per-configuration outputs (Mx.json, Mx_review.md) and evaluations
results/tables/        CSV + LaTeX tables, significance tests
results/figures/       PNG figures
results/benchmark.json computed results served to the web app's Ablation Benchmark tab
```
