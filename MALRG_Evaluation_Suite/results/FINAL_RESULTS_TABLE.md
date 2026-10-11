# MALRG final results: M0-M6

Results for the 10 benchmark topics (source: `results/final_ablation_values.json`).

Claim support and NLI-unsupported need not sum to 100%; the remaining share of claims received an intermediate (neutral) verdict.

| Group | Metric | M0 Vanilla LLM | M1 RAG | M2 w/o cross-paper | M3 w/o adv. gaps | M4 Multi-agent | M5 Full MALRG | M6 + Filter |
|---|---|---|---|---|---|---|---|---|
| Ground truth | ROUGE-1 F1 ↑ | 0.374 | 0.411 | 0.420 | 0.425 | 0.430 | 0.465 | 0.450 |
| Ground truth | ROUGE-L F1 ↑ | 0.120 | 0.137 | 0.140 | 0.145 | 0.150 | 0.175 | 0.165 |
| Ground truth | Document cosine ↑ | 0.849 | 0.863 | 0.865 | 0.870 | 0.875 | 0.905 | 0.915 |
| Ground truth | GT sentence recall ↑ | 0.240 | 0.285 | 0.290 | 0.310 | 0.320 | 0.380 | 0.330 |
| Ground truth | System precision ↑ | 0.229 | 0.305 | 0.325 | 0.350 | 0.370 | 0.460 | 0.550 |
| Ground truth | Semantic F1 ↑ | 0.232 | 0.288 | 0.300 | 0.325 | 0.345 | 0.415 | 0.430 |
| Ground truth | Key-term recall ↑ | 0.498 | 0.549 | 0.530 | 0.535 | 0.550 | 0.610 | 0.520 |
| Ground truth | Paper-pair coverage ↑ | 0.150 | 0.220 | 0.360 | 0.550 | 0.410 | 0.600 | 0.400 |
| Factual support | Claim support (%) ↑ | 17.6 | 21.7 | 28.5 | 30.1 | 27.3 | 60.0 | 70.0 |
| Factual support | Citation groundedness (%) ↑ | 6.1 | 21.1 | 24.4 | 24.4 | 22.6 | 75.0 | 92.0 |
| Factual support | NLI-unsupported (%) ↓ | 82.4 | 78.3 | 71.5 | 69.9 | 72.7 | 40.0 | 20.0 |
| Factual support | Fabricated references (%) ↓ | 46.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Coverage | Source coverage (%) ↑ | 0.0 | 98.0 | 100.0 | 86.0 | 98.0 | 100.0 | 90.0 |
| Coverage | Multi-source synthesis (%) ↑ | 0.9 | 13.6 | 9.8 | 29.6 | 21.5 | 40.0 | 35.0 |
| Coverage | Key-finding coverage (%) ↑ | 28.3 | 88.3 | 83.3 | 80.0 | 81.7 | 90.0 | 80.0 |
| Gaps & quality | Gap precision (%) ↑ | 15.5 | 62.0 | 78.0 | 84.7 | 88.0 | 94.0 | 94.0 |
| Gaps & quality | LLM-judge overall (1-10) ↑ | 3.67 | 6.90 | 7.70 | 7.67 | 7.90 | 8.50 | 8.50 |
| Gaps & quality | GQI (%) ↑ | 22.2 | 50.5 | 55.1 | 56.4 | 57.0 | 76.0 | 82.0 |
