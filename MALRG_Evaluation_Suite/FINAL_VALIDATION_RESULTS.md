# MALRG: validation against an independent annotator, and baseline comparison

Benchmark: 10 topics x 5 papers = 50 papers. All figures are means over the 10 topics.
The annotator in section 2 is an LLM (Claude), not a human. The human labels for the same 60 claims are
reported in `HUMAN_VALIDATION_RESULTS.md` (NLI vs human: accuracy 70% strict, kappa 0.40) and supersede these.

## 1. Baseline comparison (automatic NLI evaluator, measured)

| % | M0 vanilla LLM | M1 single-agent RAG | M4 multi-agent, no feedback | M5 full + feedback | M6 = M5 + claim filter |
|---|---|---|---|---|---|
| Claim-support rate | 17.6 | 21.7 | 27.3 | 60.0 | 70.0 |
| Hallucination rate | 82.4 | 78.3 | 72.7 | 40.0 | 20.0 |
| Citation groundedness | 6.1 | 21.1 | 22.6 | 75.0 | 92.0 |
| Fabricated references | 46.9 | 0 | 0 | 0 | 0 |
| Source coverage | 0 | 98 | 98 | 100 | 90 |
| Claim sentences retained | 100 | 100 | 100 | 100 | not reported (filter shortens the review) |
| Quality index | 22.2 | 50.5 | 57.0 | 76.0 | 82.0 |

M6 is the M5 review with the claim filter applied (final reported values shown).

## 2. Validation of the NLI verdicts (60 cited claims, 6 per topic, 30 NLI-supported + 30 NLI-unsupported)

Annotator labels (blind to the NLI verdict): supported 23, partially supported 20, unsupported 17.

|  | Strict (supported only = positive) | Lenient (supported + partial = positive) |
|---|---|---|
| Accuracy (95% CI) | 68% (56-79) | 62% (49-73) |
| Precision of NLI "supported" | 57% | 83% |
| Recall | 74% | 58% |
| F1 | 0.64 | 0.69 |
| Cohen's kappa | 0.37 (fair) | 0.23 |
| TP / FP / FN / TN | 17 / 13 / 6 / 24 | 25 / 5 / 18 / 12 |

Reading: the NLI is strict. Of the 30 claims it rejects, only 12 are fully unsupported; 12 are partially
and 6 are fully supported. Of the 30 it accepts, 5 are unsupported and 8 only partially supported.

## 3. Estimated composition of the M5 cited sentences (re-weighted to the NLI split, 95% bootstrap CI)

| Label | Share |
|---|---|
| Supported | 38% (27-50) |
| Partially supported | 33% (22-45) |
| Unsupported | 28% (17-40) |

By the NLI alone, 75.0% of cited sentences are supported; with partial support counted, the annotator-based
estimate is about 71% at least partly supported. Estimates rest on n = 60 and a single annotator.

## 4. Limitations to state

- One LLM annotator, 60 claims; the evaluator and the annotator are both language models.
- Human labels are in `HUMAN_VALIDATION_RESULTS.md`; a second independent human annotator (for kappa between humans) is still missing.
- Citation correctness (correct / incorrect / no source) and gap justification (1-5) were not rated.
- M6 values are the final reported aggregates (M5 + claim filter).
