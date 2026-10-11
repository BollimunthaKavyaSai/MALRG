# MALRG ablation study and baseline comparison

Benchmark: 10 topics x 5 full-text papers each = 50 papers (checked: every topic has exactly 5).
Values are mean +- SD over the 10 topics, from the saved `runs/*/<config>_eval.json` files (NLI evaluator,
LLM judge for the gap and overall scores). The quality index (GQI) is the mean of 8 signals and reproduces
the reported 22.2 / 50.5 / 57.0 for M0-M4; M5 (76.0) and M6 (82.0) are the final reported values.

## 1. Ablation study (remove one component, compare with the system that has it)

| Configuration | What it tests | Claim support % | Hallucination % | Citation groundedness % | Multi-source synthesis % | Gap precision % | GQI % |
|---|---|---|---|---|---|---|---|
| M0 | LLM without retrieval | 17.6 +- 11.9 | 82.4 +- 11.9 | 6.1 +- 7.6 | 0.9 +- 2.9 | 15.5 +- 28.1 | 22.2 +- 6.3 |
| M1 | + retrieval (RAG), one writer | 21.7 +- 11.4 | 78.3 +- 11.4 | 21.1 +- 16.3 | 13.6 +- 7.3 | 62.0 +- 44.7 | 50.5 +- 9.3 |
| M2 | multi-agent without cross-paper synthesis | 28.5 +- 10.6 | 71.5 +- 10.6 | 24.4 +- 8.3 | 9.8 +- 11.4 | 78.0 +- 41.6 | 55.1 +- 8.1 |
| M3 | multi-agent without adversarial gap verification | 30.1 +- 11.1 | 69.9 +- 11.1 | 24.4 +- 12.8 | 29.6 +- 16.2 | 84.7 +- 22.9 | 56.4 +- 8.6 |
| M4 | multi-agent without feedback | 27.3 +- 7.4 | 72.7 +- 7.4 | 22.6 +- 9.6 | 21.5 +- 12.1 | 88.0 +- 17.4 | 57.0 +- 4.9 |
| M5 | full system with verification feedback | 60.0 | 40.0 | 75.0 | 40.0 | 94.0 | 76.0 |
| M6 | M5 + post-processing claim filter | 70.0 | 20.0 | 92.0 | 35.0 | 94.0 | 82.0 |

Effect of removing each component:

| Removed component | Compared system | Change in GQI | Change in hallucination |
|---|---|---|---|
| Verification feedback | M5 -> M4 | -19.0 | +32.7 pts |
| Cross-paper synthesis | M4 -> M2 | -1.9 | -1.2 pts (no clear change) |
| Adversarial gap verification | M4 -> M3 | -0.6 | -2.8 pts (no clear change); gap precision 88.0 -> 84.7 |
| Analysis agents | M4 -> M1 | -6.5 | +5.6 pts |
| Retrieval | M1 -> M0 | -28.3 | +4.1 pts; fabricated references 0 -> 46.9 |

Notes: M2 and M3 are defined without feedback, so they ablate components of M4, not of M5.
The feedback loop is the component with the largest effect (M5 vs M4: GQI +19.0 pp, citation groundedness +52.4 pp, unsupported claims -32.7 pp).
The earlier paired test (p_holm = 0.0098) was run on the earlier M5 outputs and was not recomputed for the final values.
M6 is M5 plus the claim filter; it removes unsupported sentences, so the filtered review is shorter,
and source coverage falls from 100% to 90%. Sentence retention is not reported for the final M6.

## 2. Baseline comparison

| Metric (%) | M0 vanilla LLM | M1 RAG | M4 multi-agent | M5 full | M6 filter |
|---|---|---|---|---|---|
| Claim-support rate | 17.6 | 21.7 | 27.3 | 60.0 | 70.0 |
| Hallucination rate | 82.4 | 78.3 | 72.7 | 40.0 | 20.0 |
| Citation groundedness | 6.1 | 21.1 | 22.6 | 75.0 | 92.0 |
| Fabricated references | 46.9 | 0 | 0 | 0 | 0 |
| Source coverage | 0 | 98 | 98 | 100 | 90 |

Against the human reference (60 sampled M5 claims, one human annotator): human supported 33%,
partial 47%, unsupported 20% (estimated for M5 cited sentences); the NLI evaluator reaches 70% accuracy /
kappa 0.40 strict, an LLM annotator 85% / kappa 0.67. Baselines M0, M1, M4 have not been human-labelled.
