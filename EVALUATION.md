# Evaluation, Ablation Study & Feedback Loop

This document explains how MALRG is evaluated **automatically, without manual labels**, how the
ablation study is run, what the feedback loop does, and how the numerical analysis is produced.
Every number shown in the *Ablation Benchmark* tab and in `results/ablation/` comes from the code
described here. Nothing is hard-coded.

---

## 1. Quick start

```bash
# 0. one-time: Python deps are in requirements.txt; GROQ_API_KEY must be set in .env

# 1. build the frozen benchmark (OpenAlex + full text, no LLM calls)
python -m evaluation.ablation build

# 2. generate reviews for all configurations M0..M5 (resumable)
python -m evaluation.ablation generate

# 3. compute the metrics (local NLI + Qwen judge)
python -m evaluation.ablation evaluate

# 4. tables, statistics, charts, report
python -m evaluation.ablation analyze

# or everything in order:
python -m evaluation.ablation all

# progress overview
python -m evaluation.ablation status
```

**Groq free tier.** Each model is limited to about 8K tokens/minute and has a daily token quota.
Every stage is cached under `benchmark/<topic>/`, so when the quota runs out the runner stops
cleanly. Re-run the same command later and it continues where it stopped. Team members can split
the topics (`--topics 0 1 2`, `--topics 3 4 5`, …), each using their own key, and then copy their
`benchmark/<topic>/` folders together before `analyze`.

Approximate cost per topic (5 papers): about 90K tokens on `gpt-oss-20b` (paper analysis, gap
agents, taxonomy, fact-checker), about 45K on `gpt-oss-120b` (cross-paper agent, writer, reviser)
and about 60K on the Qwen judge.

---

## 2. Benchmark (frozen inputs)

`evaluation/config.py` lists **10 topics** from different domains (multi-agent LLMs, medical
multimodal LLMs, autonomous vehicles, plant disease, smart agriculture, fraud-detection GNNs, RAG,
federated learning, credit-risk XAI, tutoring systems).

For each topic `evaluation/benchmark.py`:

1. searches OpenAlex for open-access, peer-reviewed articles/reviews from 2020–2025 that have an
   abstract, excluding preprint servers;
2. ranks candidates with the project's own hybrid ranker
   (0.60 MiniLM semantic + 0.20 TF-IDF + 0.15 title relevance + 0.05 metadata quality);
3. keeps the top 5 papers and extracts their full text (publisher PDF, other open-access
   locations, or the official Europe PMC full-text API). If no full text is available, it falls
   back to the abstract.

The paper set is written to `benchmark/<topic>/papers.json` and **shared by every
configuration**, so differences between configurations come only from the agents.

> The local MongoDB `malrg.paper` collection created by `server/seed_corpus.js` contains
> template-generated papers. Do **not** use it for evaluation. The benchmark uses real OpenAlex
> papers instead.

---

## 3. Ablation configurations

| ID | Configuration | What is removed |
|---|---|---|
| **M0** | Vanilla LLM | Everything: one prompt, no retrieval. The model writes its own references. |
| **M1** | Retrieval + Writer | Full-text analysis, cross-paper agent, gap agents. The writer sees only abstracts. |
| **M2** | w/o Cross-Paper | The cross-paper synthesis agent. The gap agents also run without it. |
| **M3** | w/o Skeptic/Verifier | The adversarial gap verification. Proposer gaps are accepted as they are. |
| **M4** | Full pipeline | Nothing (no feedback loop). |
| **M5** | Full + Feedback (**proposed**) | Nothing. Adds Critic → Fact-Checker → Reviser. |

Design choices that keep the comparison clean:

- Every configuration uses the **same writer model and prompt** (`gpt-oss-120b`). M0 uses the
  same output format.
- **M3 gets exactly the same proposer candidates as M4.** Only the Skeptic/Verifier decision
  differs.
- **M5 starts from exactly the M4 draft**, so M5 − M4 is the effect of the feedback loop alone.
- The runner sets `MALRG_STRICT_LLM=1`. If the LLM is unavailable, a stage fails and is retried
  later instead of silently falling back to template text, so fallback text never enters the
  results.

The same switches exist in the live app: `ablation_mode` on `POST /api/review/generate`. The
default is `M5_full_feedback`.

---

## 4. The feedback loop ("adding feedback")

Implemented in `review_feedback.py` as the `feedback_agent` node after the writer in the
LangGraph pipeline (`app.py`).

Each round:

1. **Citation Auditor** (deterministic) flags:
   - invalid citation numbers;
   - papers that are never cited;
   - factual sentences without a citation;
   - research gaps without a citation;
   - over-certain wording;
   - too few cross-paper comparison sentences.
2. **Fact-Checker**: a MiniLM2 NLI model pre-screens every cited sentence against passages
   retrieved from the cited paper. Sentences below 0.85 entailment go to an LLM fact-checker
   (`gpt-oss-20b`), which labels them SUPPORTED / PARTIAL / UNSUPPORTED / CONTRADICTED and
   attaches the evidence.
3. **Reviser** (writer model) fixes only the flagged sentences, using only the shown evidence.
4. The revision is **kept only if the critic score increases**, so the loop can only improve
   the review. It runs up to 2 rounds (`MALRG_FEEDBACK_ROUNDS`).

**Human feedback.** The same Reviser powers `POST /api/review/feedback/{task_id}` (Python) and
`POST /api/review/:id/feedback` (Node), shown as the *Reviewer feedback* box under the manuscript.
The user's comments are applied together with the automatic findings, and the UI shows the critic
score before and after.

**No self-grading.** The feedback loop uses MiniLM2-NLI and `gpt-oss-20b`. The evaluator uses
**different** models (DeBERTa-v3 NLI and the Qwen judge), so the system is not optimized against
the judge that scores it.

---

## 5. Metrics (all automatic)

Implemented in `evaluation/metrics.py`.

**Evidence policy:** claims are checked only against **primary source text** (abstract +
extracted full text), never against the system's own LLM notes. Each reference in a review is
resolved as follows:

- the title matches a benchmark paper → that paper's text;
- the DOI or title is found in OpenAlex → the real paper's abstract;
- otherwise → *unverifiable*.

A fabricated reference therefore can never support a claim.

### A. Citation integrity
| Metric | Definition |
|---|---|
| Reference Validity ↑ | % of reference-list entries that resolve to a real publication |
| Citation Validity ↑ | % of in-text `[n]` markers that point to a valid reference |
| Source Coverage ↑ | % of the topic's benchmark papers cited in the body (slide 28: papers used / retrieved) |
| Citation Density ↑ | % of sentences that carry a citation |

### B. Grounding / hallucination
| Metric | Definition |
|---|---|
| Citation Groundedness ↑ | % of sampled cited claims the judge labels SUPPORTED by the cited paper's text |
| Groundedness (lenient) ↑ | Same, with PARTIAL counted as 0.5 |
| Hallucination Rate ↓ | % of sampled claims labeled UNSUPPORTED or CONTRADICTED (uncited claims are checked against all sources) |
| Contradiction Rate ↓ | % of claims contradicted by the evidence |
| NLI Entailment ↑ | % of **all** cited sentences entailed (p ≥ 0.5) by the cited paper, using local `cross-encoder/nli-deberta-v3-base` |
| Judge–NLI agreement | How often the two independent checkers agree (evaluator reliability) |

The judge (`qwen/qwen3.8-27b`, a different model family from the GPT-OSS generators) sees up to
18 sentences per review. All gap sentences (up to 5) are included, and the rest are a seeded
random sample. Evidence passages are retrieved with MiniLM and prefixed with the paper's authors
and year.

### C. Cross-paper synthesis
| Metric | Definition |
|---|---|
| Cross-Paper Synthesis ↑ | % of cited sentences that cite ≥ 2 different valid papers |
| Grounded comparisons, per 1k words ↑ | Multi-citation sentences with comparative wording ("whereas", "in contrast", …) |
| Paper-Pair Coverage ↑ | Share of cited paper pairs that are compared at least once |

### D. Research gaps
| Metric | Definition |
|---|---|
| Gap Grounding ↑ | Judge support of gap statements (SUPPORTED + 0.5 × PARTIAL) |
| Gap Specificity ↑ | Judge: the gap names a concrete method / task / dataset / setting |
| Gap Novelty ↑ | Judge: no supplied paper already resolves the gap |
| Gap Citation Rate ↑ | % of gap sentences with citations |
| **Skeptic evaluation** | Every proposed and every verified gap is judged. *Precision without Skeptic* (M3), *precision after Skeptic + Verifier* (M4), *Skeptic precision* (% of rejected gaps that were truly unsupported) and rejection rate |

### E. Structure & efficiency
The project's deterministic evaluator (`evaluate_generated_review`) covers structure, word count,
truncation and redundancy (structure score 0–100). It is complemented by the over-claim count
(e.g. "definitively demonstrates") and by LLM tokens, LLM calls and generation time per
configuration.

### Overall Quality Index (OQI)
OQI is the mean of: Reference Validity, Citation Validity, Groundedness, 1 − Hallucination Rate,
Source Coverage, min(1, Synthesis / 0.30), Gap Grounding, and Structure / 100.

### What is *not* claimed
P@K, Recall@K, nDCG and "Evidence F1" need human relevance or extraction labels. They are **not**
reported, because they cannot be computed honestly without manual validation. The retrieval
section reports the hybrid-ranker scores of the selected papers (`benchmark/<topic>/topic.json`).

---

## 6. Numerical analysis (`evaluation/analysis.py`)

- Mean ± SD and **95% bootstrap confidence intervals** over topics, using only topics that have
  every configuration (paired design).
- A **paired Wilcoxon signed-rank test** compares M5 against every other configuration, and M4
  against each of its ablations. p-values are **Holm-corrected** within each metric. Effect size
  is the **matched-pairs rank-biserial correlation**, reported with win/tie/loss counts.
- **Component contributions:** the drop in each metric when a component is removed
  (M4−M0, M4−M1, M4−M2, M4−M3, M5−M4).
- **Robustness:** NLI entailment at thresholds 0.3 / 0.5 / 0.7 / 0.9.
- **Feedback analysis:** the critic score per round, and how many revisions were accepted.

Outputs are written to `results/ablation/`:

- `REPORT.md`
- `ablation_table.md` / `.tex`
- `per_review_metrics.csv`
- `significance.csv`
- `component_contributions.csv`
- `gap_verification.csv`
- `feedback_rounds.csv`
- `nli_threshold_sensitivity.csv`
- `summary.json` (read by the dashboard)
- `charts/*.png`

---

## 7. Files

| File | Role |
|---|---|
| `llm_client.py` | Groq client: keys from `.env`, retries, quota detection, token accounting |
| `grounding.py` | Review parsing, citation extraction, evidence index, NLI scorer |
| `claim_checking.py` | Batched LLM claim checking (used by the fact-checker and the judge) |
| `review_feedback.py` | Critic → Fact-Checker → Reviser loop, human feedback |
| `evaluation/benchmark.py`, `evaluation/openalex.py` | Frozen benchmark construction, reference verification |
| `evaluation/ablation.py` | Resumable runner for M0–M5 |
| `evaluation/metrics.py` | All metrics |
| `evaluation/analysis.py` | Statistics, tables, charts, report |
