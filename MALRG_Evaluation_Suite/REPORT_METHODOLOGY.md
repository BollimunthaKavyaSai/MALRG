# MALRG: Evidence-Grounded Multi-Agent Literature Review Generation with Verification-Guided Feedback

**Team 02, Amrita School of Computing, Chennai.** Supervisor: Dr. Bharathi Mohan G.

---

## 1. Problem and contributions

LLM-based literature-review tools retrieve and summarize well, but they often (i) state claims
their cited sources do not support, (ii) invent references, (iii) describe papers one by one
instead of comparing them, and (iv) propose generic or already-solved "research gaps".
MALRG addresses these problems with an evidence-grounded multi-agent pipeline (Review-1,
slides 20–25). This work adds a fully automatic, reproducible evaluation of the pipeline.

**Contributions**

1. **Evidence-grounded multi-agent pipeline.** Query understanding, live scholarly retrieval
   with hybrid ranking, full-text PDF extraction, paper-level evidence extraction with an
   **Evidence Gate** (each extracted quote must occur verbatim in the PDF), cross-paper synthesis,
   taxonomy, citation structuring and a citation-grounded writer, orchestrated with LangGraph.
2. **Heterogeneous adversarial gap triad.** Proposer → deterministic Evidence Gate →
   **Skeptic from a different model family** (Qwen vs. GPT-OSS) → Grounded Verifier.
   Using a different model family for the Skeptic reduces "self-agreement", where a model
   accepts its own unsupported proposals.
3. **Verification-Guided Iterative Refinement (VGIR), the feedback loop (novel).** After writing,
   a local NLI critic verifies every sentence against the full text of the paper it cites. The loop:
   - **re-cites deterministically** when a different source entails the claim (no LLM call);
   - sends the remaining unsupported, uncited or overclaimed sentences, with their evidence windows,
     to a reviser in batches;
   - accepts a rewrite **only if it then passes verification** (verified-acceptance guard);
   - adds verified sentences for sources the review never cited.

   The critic uses a *different* NLI model than the evaluator, so the system is not optimized
   against the instrument that scores it.
4. **Reference-free automatic evaluation protocol.** It needs no human labels: NLI-based
   citation groundedness and precision, hallucination rate, fabricated-reference detection by
   OpenAlex resolution, synthesis and coverage metrics, a blind cross-family LLM judge, and
   silver labels for retrieval relevance and gap validity.
5. **Six-configuration ablation with paired statistics.** Bootstrap CIs, Wilcoxon signed-rank
   tests with Holm correction, effect sizes, Friedman/Kendall's W and component-contribution
   analysis.

## 2. System architecture (as implemented)

| # | Agent | Implementation | Model |
|---|---|---|---|
| 1 | Query Understanding | intent, core concepts, 4 expanded queries | qwen3.8-27b |
| 2 | Scholarly Retrieval | live OpenAlex (open-access articles) + arXiv API | – |
| 3 | Hybrid Ranking | 0.60 MiniLM semantic + 0.20 TF-IDF + 0.15 title + 0.05 quality | all-MiniLM-L6-v2 |
| 4 | PDF Extraction | PyMuPDF full text, section detection, bibliography removal, 110-word passages | – |
| 5 | Paper Analysis + Evidence Gate | problem, data, method, results, metrics, limitations, contributions, verbatim quotes | gpt-oss-20b |
| 6 | Taxonomy | agglomerative clustering of paper embeddings | MiniLM |
| 7 | Cross-Paper Synthesis | themes, agreements, contradictions, trade-offs, repeated limitations | gpt-oss-20b |
| 8 | Gap Triad | Proposer → Evidence Gate (embedding link ≥ 0.35) → Skeptic → Verifier | gpt-oss-20b / **qwen** / gpt-oss-20b |
| 9 | Writer | one shared prompt for every configuration | gpt-oss-120b |
| 10 | Feedback (VGIR) | NLI critic + re-citation + reviser + verified acceptance | DeBERTa-v3 NLI + gpt-oss-120b |
| 11 | Citation Structuring | IEEE references generated from verified metadata (never by the LLM) | – |

## 3. Experimental design

**Benchmark.** 10 topics from diverse domains: RAG and hallucination, plant disease, federated
learning in healthcare, XAI for autonomous driving, GNNs for drug discovery, multimodal LLMs for
diagnosis, IoT agriculture, LLM multi-agent systems, transformers for time series, and AI in education.
For each topic, the top-ranked papers with downloadable full text (at most one survey) are
**frozen** (5 per topic, 50 full-text papers), so every configuration sees exactly the same sources.

**Ablation configurations** (Review-1, slide 26, plus the feedback extension):

| ID | Configuration | Retrieval | Evidence analysis | Cross-paper | Gap triad | Feedback |
|---|---|---|---|---|---|---|
| M0 | Vanilla LLM | – | – | – | – | – |
| M1 | Retrieval + Writer (abstracts) | ✓ | – | – | – | – |
| M2 | w/o Cross-Paper Synthesis | ✓ | ✓ | – | ✓ | – |
| M3 | w/o Adversarial Verification (Proposer only) | ✓ | ✓ | ✓ | – | – |
| M4 | Full Multi-Agent Pipeline (Review-1 system) | ✓ | ✓ | ✓ | ✓ | – |
| M5 | **M4 + Verification-Guided Feedback (proposed)** | ✓ | ✓ | ✓ | ✓ | ✓ |

**Additional experiments**

| Group | Configurations | Question answered |
|---|---|---|
| Baseline models | M0 (vanilla GPT-OSS-120B), B1 (vanilla GPT-OSS-20B), B2 (vanilla Qwen3-27B), M1 (RAG baseline) vs. M5 | Is the gain due to the architecture or just the choice of LLM? |
| Prompt comparison | P1 structured rules, P2 zero-shot minimal, P3 expert persona + reviewer criteria, P4 plan-then-write, P5 evidence-anchored constraints | Which writer prompt yields the most grounded, best-synthesized review on *identical* evidence? (single writer: GPT-OSS-20B) |
| User-feedback learning | U1 = M4 + learned preferences vs. M4 | Does feedback given once get applied automatically the *next time*, on unseen topics? |

**User-feedback learning protocol.**
1. Five realistic feedback messages are given once, after the review for topic 1: a length limit,
   a "Datasets and Evaluation Metrics" section, exact numeric results, no first-person language,
   and a "Key Takeaways" section.
2. A preference agent distills them into persistent imperative rules (newer feedback overrides
   conflicting rules).
3. The rules are injected into every later writer prompt.
4. Compliance is checked automatically on the 9 *held-out* topics, before (M4) and after (U1),
   while quality metrics confirm that following the preferences does not reduce grounding.

The same mechanism is available to real users:
- `python run_all.py feedback "..."` records feedback.
- `python run_all.py review "<topic>"` generates with the learned preferences.

**Controls**
- Same writer model and prompt in all configurations.
- Temperature 0 and a fixed seed.
- Every LLM call is cached on disk, so the study is exactly reproducible.
- M5 starts from the identical M4 draft, so M5 − M4 isolates the feedback loop.
- Identical post-processing for all configurations.

## 4. Evaluation metrics (all automatic)

Sentence-level verification uses `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Each claim
(with its citations stripped) is scored against the best passage and against an evidence window
of the top-3 passages of each cited paper, retrieved by MiniLM. A claim counts as **supported**
when P(entailment) ≥ 0.5.

| Metric | Definition |
|---|---|
| Citation Groundedness (CG) | cited sentences entailed by their cited source(s) / cited sentences |
| Citation Precision (CP) | citations that individually (or jointly) entail their sentence / all citations |
| Hallucination Rate (HR) | claim sentences entailed by **no** passage of any available source / claim sentences |
| Fabricated Reference Rate (FRR) | references that do not resolve on OpenAlex (M0) or invalid source ids (M1–M5) |
| Uncited Claim Rate | factual-looking sentences without a citation / claim sentences |
| Source Coverage | frozen benchmark papers cited at least once / 5 |
| Multi-Source Synthesis | cited sentences integrating ≥ 2 distinct sources |
| Cross-Paper Comparison | sentences with a comparative marker **and** ≥ 2 sources |
| Judge scores (1–10) | blind G-Eval rubric by qwen3.8-27b: coverage, synthesis, critical analysis, gap quality, coherence, faithfulness |
| Verified Gap Precision | gaps judged grounded ∧ specific ∧ novel against the source limitations / gaps stated |
| Judge Claim Support | independent LLM check of 6 sampled citation claims (triangulates the NLI metric) |
| Skeptic precision (silver) | rejected candidate gaps that the independent judge also deems invalid |
| Retrieval P@5, nDCG@10, MRR | against graded (0/1/2) silver relevance labels for 4 rankers |
| Evidence Fidelity | extracted evidence quotes found verbatim in the PDF |
| Review-1 slide-28 indicators | synthesis indicators, gap indicators, structure, certainty flags, word count, redundancy |
| Preference Compliance | share of the 5 learned preferences satisfied (each checked by a deterministic rule) |
| **Grounded Quality Index (GQI)** | unweighted mean of CG, CP, 1−HR, coverage, multi-source synthesis, gap precision, judge overall/10, structure |

## 5. Numerical analysis

The additional experiments use the same machinery:
- **Baselines:** M5 is compared with each baseline (Wilcoxon + Holm).
- **Prompts:** a Friedman test across the 5 prompts with mean ranks, then the best prompt is compared
  with each of the others (Wilcoxon + Holm).
- **Preferences:** per-rule compliance before and after, overall compliance (paired Wilcoxon), and
  quality side-effects.

For the ablation study:

For each metric and configuration:
- mean ± SD and a 95% bootstrap CI (10,000 resamples) over topics;
- a paired **Wilcoxon signed-rank** test of M5 against every other configuration, with
  Holm–Bonferroni correction;
- paired effect sizes (Cohen's d_z and rank-biserial r) and the win rate across topics;
- a **Friedman** omnibus test with Kendall's W.

**Component contribution** is the paired Δ when a component is removed:

| Component | Δ |
|---|---|
| Retrieval grounding | M1 − M0 |
| Evidence analysis and gaps | M2 − M1 |
| Cross-paper synthesis | M4 − M2 |
| Adversarial verification | M4 − M3 |
| Feedback | M5 − M4 |

## 6. Threats to validity

- **The NLI verifier is conservative.** Abstractive synthesis sentences that combine or
  interpret evidence are often scored "neutral", so absolute CG values are a lower bound. The
  comparison is still fair because the same verifier and thresholds apply to every
  configuration, and the blind LLM judge and the LLM claim check give an independent
  second view.
- **Feedback and metric coupling.** VGIR uses an NLI critic, but a different NLI model from
  the evaluator. Gains are also cross-checked with the LLM judge.
- **Silver labels.** Retrieval relevance and gap validity come from an LLM judge. They are
  reported as silver labels, not human ground truth.
- **Scale.** 10 topics × 5 papers, limited by the free-tier API quota. The paired design and
  non-parametric tests are chosen for this sample size.
- **LLM-judge bias.** It is reduced by blind evaluation and by using a different model family
  from the writer.
