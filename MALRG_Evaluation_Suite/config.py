"""
Central configuration for the MALRG evaluation suite.

Every experiment parameter lives here so that the full study can be
reproduced from a single file. Nothing in the results is hand-entered:
all numbers are produced by `run_all.py` from the logged runs.
"""

import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
PDF_DIR = DATA_DIR / "pdfs"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = ROOT / "cache" / "llm"

for _d in (BENCHMARK_DIR, PDF_DIR, RUNS_DIR, TABLES_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark topics (diverse domains so results are not topic-specific)
# ---------------------------------------------------------------------------
TOPICS = [
    "Retrieval-augmented generation for reducing hallucination in large language models",
    "Deep learning for plant disease detection",
    "Federated learning for healthcare data privacy",
    "Explainable artificial intelligence for autonomous driving",
    "Graph neural networks for drug discovery",
    "Multimodal large language models for medical diagnosis",
    "Internet of Things and machine learning for smart agriculture",
    "Large language model based multi-agent systems",
    "Transformer models for time series forecasting",
    "Artificial intelligence in education and intelligent tutoring systems",
]

PAPERS_PER_TOPIC = 5          # frozen source set per topic (same for every config)
START_YEAR = 2017
END_YEAR = 2026
CANDIDATES_PER_QUERY = 25     # OpenAlex + arXiv candidates per expanded query
MAX_SURVEYS_PER_TOPIC = 1     # the rest must be primary studies
MIN_FULLTEXT_WORDS = 1500     # a paper must have real full text to be selected

# Hybrid ranking weights (as presented in Review-1, slide 22)
RANK_WEIGHTS = {"semantic": 0.60, "tfidf": 0.20, "title": 0.15, "quality": 0.05}

# ---------------------------------------------------------------------------
# Models (Groq). Roles are split across model families on purpose:
#  * the writer is identical for every ablation config (fair comparison)
#  * the skeptic and the evaluation judge come from a different model family
#    than the proposer / writer, reducing self-agreement and self-preference bias
# ---------------------------------------------------------------------------
MODELS = {
    "query": "qwen/qwen3.8-27b",
    "analysis": "openai/gpt-oss-20b",
    "cross_paper": "openai/gpt-oss-20b",
    "gap_proposer": "openai/gpt-oss-20b",
    "gap_skeptic": "qwen/qwen3.8-27b",
    "gap_verifier": "openai/gpt-oss-20b",
    "writer": "openai/gpt-oss-120b",
    "reviser": "openai/gpt-oss-120b",
    "judge": "qwen/qwen3.8-27b",
}

TOKENS_PER_MINUTE = 7600      # stay just under Groq's 8K TPM free-tier limit
TEMPERATURE = 0.0
SEED = 42

# ---------------------------------------------------------------------------
# Verification (local NLI, no LLM, no human labels)
# ---------------------------------------------------------------------------
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_ENTAIL_THRESHOLD = 0.50   # P(entailment) needed to call a claim supported
NLI_CONTRA_THRESHOLD = 0.60   # P(contradiction) needed to flag a contradiction
NLI_TOP_K_PASSAGES = 4        # passages per cited paper sent to the NLI model
NLI_WINDOW_PASSAGES = 3       # top passages concatenated into one evidence window (<=512 tokens)
CHUNK_WORDS = 110
CHUNK_OVERLAP = 30

# ---------------------------------------------------------------------------
# Feedback loop (Verification-Guided Iterative Refinement)
# ---------------------------------------------------------------------------
FEEDBACK_MAX_ROUNDS = 3
FEEDBACK_MAX_SENTENCES_PER_ROUND = 36
FEEDBACK_MAX_DELETE_FRACTION = 0.15  # deletion budget (share of draft sentences)
FEEDBACK_BATCH_SIZE = 12          # flagged sentences per reviser call

# ---------------------------------------------------------------------------
# Ablation configurations (Review-1 slide 26 + the feedback extension)
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "M0": {"name": "M0: Vanilla LLM (no retrieval)", "retrieval": False, "analysis": False,
           "cross_paper": False, "gap_triad": False, "gaps": False, "feedback": False},
    "M1": {"name": "M1: Retrieval + Writer", "retrieval": True, "analysis": False,
           "cross_paper": False, "gap_triad": False, "gaps": False, "feedback": False},
    "M2": {"name": "M2: w/o Cross-Paper Synthesis", "retrieval": True, "analysis": True,
           "cross_paper": False, "gap_triad": True, "gaps": True, "feedback": False},
    "M3": {"name": "M3: w/o Adversarial Verification", "retrieval": True, "analysis": True,
           "cross_paper": True, "gap_triad": False, "gaps": True, "feedback": False},
    "M4": {"name": "M4: Full Multi-Agent Pipeline", "retrieval": True, "analysis": True,
           "cross_paper": True, "gap_triad": True, "gaps": True, "feedback": False},
    "M5": {"name": "M5: Full + Verification-Guided Feedback (Proposed)", "retrieval": True,
           "analysis": True, "cross_paper": True, "gap_triad": True, "gaps": True, "feedback": True},
}
PROPOSED = "M5"

_FULL = {"retrieval": True, "analysis": True, "cross_paper": True, "gap_triad": True, "gaps": True,
         "feedback": False}
_VANILLA = {"retrieval": False, "analysis": False, "cross_paper": False, "gap_triad": False,
            "gaps": False, "feedback": False}

# Baseline LLMs: the same vanilla task (no retrieval) with different model families/sizes.
# M0 (gpt-oss-120b) and M1 (RAG baseline) are reused from the ablation group.
BASELINE_CONFIGS = {
    "B1": {"name": "B1: Vanilla GPT-OSS-20B", **_VANILLA, "writer_model": "openai/gpt-oss-20b"},
    "B2": {"name": "B2: Vanilla Qwen3-27B", **_VANILLA, "writer_model": "qwen/qwen3.8-27b"},
}
BASELINE_COMPARISON = ["M0", "B1", "B2", "M1", "M5"]

# Prompt comparison: identical full-pipeline evidence, 5 writer prompts, one writer model.
PROMPT_WRITER_MODEL = "openai/gpt-oss-20b"   # separate quota from the ablation writer
_PROMPT_NAMES = {"P1": "P1: Structured rules (default)", "P2": "P2: Zero-shot minimal",
                 "P3": "P3: Expert persona + reviewer criteria", "P4": "P4: Plan-then-write",
                 "P5": "P5: Evidence-anchored constraints"}
PROMPT_CONFIGS = {
    pid: {"name": name, **_FULL, "prompt": pid, "writer_model": PROMPT_WRITER_MODEL}
    for pid, name in _PROMPT_NAMES.items()
}
# The prompt study runs on all topics, like every other experiment.

PROMPT_TOPICS = TOPICS          # all 10 topics (statistical power)

# User-feedback learning: M4 + preference rules distilled from simulated feedback on topic 1,
# evaluated on the other (held-out) topics against M4 without preferences.
PREFERENCE_CONFIGS = {
    "U1": {"name": "U1: M4 + learned user preferences", **_FULL, "preferences": True},
}

CONFIGS = {**ABLATION_CONFIGS, **BASELINE_CONFIGS, **PROMPT_CONFIGS, **PREFERENCE_CONFIGS}
GROUPS = {
    "ablation": list(ABLATION_CONFIGS),
    "baselines": list(BASELINE_CONFIGS),
    "prompts": list(PROMPT_CONFIGS),
    "preferences": list(PREFERENCE_CONFIGS),
}
# configs whose cited claims also get the (more expensive) LLM claim check
CLAIM_JUDGE_CONFIGS = set(ABLATION_CONFIGS) | set(BASELINE_CONFIGS)

STAT_ALPHA = 0.05
BOOTSTRAP_SAMPLES = 10000
