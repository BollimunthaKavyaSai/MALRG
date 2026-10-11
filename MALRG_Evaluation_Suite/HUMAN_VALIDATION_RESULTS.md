# Human validation of the NLI verdicts (60 claims, 10 topics / 50 papers)

Source: `GT_human_labeling_BLIND_annotator_1.xlsx` and `..._annotator_2.xlsx`.
The two files contain identical labels in all 60 rows, so they are treated as ONE human annotator
(no inter-annotator agreement can be reported). An LLM (Claude) labelled the same 60 claims blind;
44 of 60 labels agree with the human (kappa 0.60), 16 were changed by the human.

Human label distribution: supported 20 (33%), partially supported 28 (47%), unsupported 12 (20%).

## NLI evaluator vs human labels

|  | Strict (supported = positive) | Lenient (supported + partial = positive) |
|---|---|---|
| Accuracy (95% CI) | 70% (57-80) | 60% (47-71) |
| Precision of NLI "supported" | 53% | 90% |
| Recall | 80% | 56% |
| F1 | 0.64 | 0.69 |
| Cohen's kappa | 0.40 (fair) | 0.20 |
| TP / FP / FN / TN | 16 / 14 / 4 / 26 | 27 / 3 / 21 / 9 |

NLI-supported claims (30): human supported 16, partial 11, unsupported 3.
NLI-unsupported claims (30): human supported 4, partial 17, unsupported 9.

## Estimated composition of M5 cited sentences (re-weighted to the NLI split, 95% bootstrap CI)

Supported 33% (23-45), partially supported 47% (35-60), unsupported 20% (10-30).
At least partly supported: about 80%.

## Limitations
One human annotator, 60 claims, evidence limited to the two shown passages; no second independent human.
