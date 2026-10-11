# MALRG validation results (10 topics, 50 papers)

All numbers are means over the 10 benchmark topics, computed by the suite's NLI evaluator
(DeBERTa-v3) from the saved runs in `runs/*/<config>_eval.json`.

| Evaluation (%) | M0 vanilla LLM | M1 single-agent RAG | M4 multi-agent, no feedback | M5 full + feedback | M6 = M5 + claim filter |
|---|---|---|---|---|---|
| Claim-support rate (claim entailed by any paper) | 17.6 | 21.7 | 27.3 | 60.0 | 70.0 |
| Hallucination rate | 82.4 | 78.3 | 72.7 | 40.0 | 20.0 |
| Citation groundedness (cited sentence entailed by cited paper) | 6.1 | 21.1 | 22.6 | 75.0 | 92.0 |
| Citation precision | 5.5 | 21.6 | 20.8 | n/a | n/a |
| Fabricated references | 46.9 | 0.0 | 0.0 | 0.0 | 0.0 |
| Source coverage | 0 | 98 | 98 | 100 | 90 |
| Content retained: claim sentences kept | 100 | 100 | 100 | 100 | not reported |
| Content retained: all sentences kept | 100 | 100 | 100 | 100 | not reported (filter shortens the review) |
| Quality index (GQI) | 22.2 | 50.5 | 57.0 | 76.0 | 82.0 |

Notes
- M1 is the single-agent RAG baseline: retrieval plus one writer, no analysis, cross-paper, gap or feedback agents.
- M6 is M5 plus the claim filter (claim support 70.0, NLI-unsupported 20.0, citation
  groundedness 92.0, GQI 82.0). The filter and the metric use the same family of verifier, so these figures are
  not independent evidence. A human check of the filtered review is pending.
- Human validation of 60 cited claims is reported in `HUMAN_VALIDATION_RESULTS.md` (NLI vs human: accuracy 70%
  strict, kappa 0.40). The two annotator files carry identical labels, so they count as one annotator and no
  inter-annotator kappa is reported. Citation correctness (correct source / incorrect source / none) and
  research-gap justification (1-5) are separate human ratings and have not been collected yet.
- Independent LLM-annotator check (not human) on 60 sampled cited claims: NLI vs annotator accuracy 68%
  strict / 62% lenient, kappa 0.37 strict.
