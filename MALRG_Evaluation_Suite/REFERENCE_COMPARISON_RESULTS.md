# Comparison with an LLM-written reference review (3 topics)

IMPORTANT: the reference reviews in `reference_reviews_llm/` were written by an LLM (Claude) from the same
5 benchmark papers per topic (abstract-level facts). They are NOT human-written. Each reference lists 6 key
findings; a system review "covers" a finding when it contains all required term groups (transparent keyword
check, see the key-finding lists in the reference files).

Topics: deep learning for plant disease detection, federated learning for healthcare data privacy,
LLM-based multi-agent systems (6 key findings each).

| Config | Plant disease | Federated learning | LLM multi-agent | Mean key-finding coverage |
|---|---|---|---|---|
| M0 vanilla LLM | 0% | 50% | 0% | 17% |
| M1 single-agent RAG | 67% | 83% | 100% | **83%** |
| M2 no cross-paper | 50% | 67% | 100% | 72% |
| M3 no adversarial gaps | 67% | 67% | 100% | 78% |
| M4 no feedback | 50% | 67% | 100% | 72% |

Reading (3-topic table; M5 and M6 are covered by the 10-topic table below)
- Retrieval is the decisive factor for covering the reference's findings (M0 17% vs 72-83%).
- Over the 10 topics M5 reaches 90.0% against 88.3% for M1, so MALRG matches or slightly exceeds the RAG baseline
  on key-finding coverage.
- The claim filter (M6) lowers coverage (80.0% on 10 topics), because it removes unsupported sentences.

Limits: 3 topics, keyword matching, LLM-written reference built from abstracts, M6 text reconstructed by
deleting flagged sentences from M5. Not a significance test.

## Update: all 10 topics (`build_reference_coverage.py`)

Key findings (6 per topic) were defined by an LLM from the benchmark abstracts for all 10 topics; full reference
prose exists for the first 3 topics only (`reference_reviews_llm/`). Same keyword rule as above.

| Config | Mean key-finding coverage (10 topics) | SD across topics |
|---|---|---|
| M0 vanilla LLM | 28.3% | 24.9 |
| M1 single-agent RAG | 88.3% | 13.7 |
| M2 no cross-paper | 83.3% | 17.6 |
| M3 no adversarial gaps | 80.0% | 18.9 |
| M4 no feedback | 81.7% | 20.0 |
| M5 full system | 90.0% | 18.9 |
| M6 M5 + claim filter | 80.0% | 27.0 |

Per-topic values are in `runs/_reference_coverage.json`. Several keywords are generic (e.g. "yield", "survey"), so
M0 gets partial credit on some topics; the check is a coarse proxy, not a semantic judgement.

### Note
Over the 10 topics the key-finding coverage is M1 88.3%, M5 90.0% and M6 80.0%, so M5 matches or slightly exceeds the plain RAG baseline, and the claim filter (M6) reduces coverage by 10 points.

The ground-truth metrics in `results/FINAL_RESULTS_TABLE.md` (ROUGE, document cosine, sentence recall, system precision,
semantic F1, key-term recall, paper-pair coverage) are computed against the human-written reviews for the 10 topics, not
against the three LLM-written files in `reference_reviews_llm/`.
