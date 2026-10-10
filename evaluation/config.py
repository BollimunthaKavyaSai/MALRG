"""Shared configuration for the benchmark, ablation runner and analysis."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(os.environ.get("MALRG_BENCHMARK_DIR", PROJECT_ROOT / "benchmark"))
RESULTS_DIR = Path(os.environ.get("MALRG_RESULTS_DIR", PROJECT_ROOT / "results" / "ablation"))

# Benchmark topics: diverse domains so results do not hinge on one field.
TOPICS = [
    "Multi-agent large language model systems for automated literature review",
    "Multimodal large language models for medical diagnosis",
    "Agentic AI for autonomous vehicles",
    "Deep learning for plant disease detection",
    "IoT-based smart agriculture monitoring",
    "Graph neural networks for financial fraud detection",
    "Retrieval-augmented generation for question answering",
    "Federated learning for privacy-preserving healthcare",
    "Explainable AI for credit risk assessment",
    "Intelligent tutoring systems using artificial intelligence in education",
]

START_YEAR = 2020
END_YEAR = 2025
PAPERS_PER_TOPIC = int(os.environ.get("MALRG_PAPERS_PER_TOPIC", "5"))

# Paper-analysis depth for benchmark runs (Groq free tier: 8K tokens/min).
ANALYSIS_MAX_CHUNKS = int(os.environ.get("MALRG_ANALYSIS_MAX_CHUNKS", "2"))

# Ablation configurations (M0 is a single-prompt baseline outside the graph).
CONFIGS = [
    ("M0_vanilla", "M0: Vanilla LLM (single prompt, no retrieval)"),
    ("M1_retrieval_only", "M1: Retrieval + Writer"),
    ("M2_no_cross_paper", "M2: w/o Cross-Paper Synthesis"),
    ("M3_no_skeptic", "M3: w/o Skeptic/Verifier"),
    ("M4_full", "M4: Full Multi-Agent Pipeline"),
    ("M5_full_feedback", "M5: Full + Feedback Loop (Proposed)"),
]
CONFIG_LABELS = dict(CONFIGS)
CONFIG_SHORT = {
    "M0_vanilla": "M0 Vanilla",
    "M1_retrieval_only": "M1 Retrieval",
    "M2_no_cross_paper": "M2 −Cross",
    "M3_no_skeptic": "M3 −Skeptic",
    "M4_full": "M4 Full",
    "M5_full_feedback": "M5 Full+FB",
}

# Independent judge model (different family from the GPT-OSS generators).
JUDGE_MODEL = os.environ.get("MALRG_JUDGE_MODEL", "qwen/qwen3.8-27b")
JUDGE_MAX_CLAIMS = int(os.environ.get("MALRG_JUDGE_MAX_CLAIMS", "18"))
JUDGE_SEED = 20261002
