"""
End-to-end ablation runner (resumable).

    python -m evaluation.ablation build      # frozen OpenAlex benchmark (no LLM)
    python -m evaluation.ablation generate   # run M0..M5 on every topic
    python -m evaluation.ablation evaluate   # automatic metrics (NLI + judge)
    python -m evaluation.ablation analyze    # tables, statistics, charts
    python -m evaluation.ablation all        # everything above, in order

Options: --topics 0 3 5   (indices into TOPICS; lets teammates split the work)
         --configs M0_vanilla M5_full_feedback
         --force           (recompute cached outputs)

Design
------
Every intermediate result is cached under benchmark/<topic>/ so the run can be
stopped (e.g. Groq daily quota) and resumed without repeating LLM calls.
Configurations share upstream stages so the ONLY difference between them is
the ablated component:

    stage              M0  M1  M2  M3  M4  M5
    retrieval (frozen)  -   x   x   x   x   x
    full-text + analysis    -   x   x   x   x
    cross-paper             -   -   x   x   x
    gap proposer            -   x*  x   x   x      (*without cross-paper input)
    skeptic + verifier      -   x*  -   x   x
    writer                  x   x   x   x   x   (M5 reuses the M4 draft)
    feedback loop           -   -   -   -   x
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

os.environ.setdefault("MALRG_STRICT_LLM", "1")  # never let fallbacks contaminate results

from .config import (  # noqa: E402
    ANALYSIS_MAX_CHUNKS,
    BENCHMARK_DIR,
    CONFIGS,
    END_YEAR,
    PROJECT_ROOT,
    RESULTS_DIR,
    START_YEAR,
    TOPICS,
)

os.chdir(PROJECT_ROOT)  # paper_cache/ and experiments/ are relative paths
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import (  # noqa: E402
    DailyQuotaExhausted,
    LLMUnavailable,
    invoke_text,
    make_llm,
    reset_token_usage,
    snapshot_token_usage,
)

from .benchmark import build_topic, topic_dir  # noqa: E402

CONFIG_IDS = [c for c, _ in CONFIGS]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

# Reads a JSON file and returns its content; if the file is missing or broken it returns the default value.
def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# Saves data as a JSON file. It creates the folder if needed and writes to a temporary file first,
# then renames it, so a crash in the middle never leaves a half-written file.
def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# Saves how long a stage took and how many LLM calls/tokens it used into the folder's stage_costs.json.
# These numbers are used later for the cost-versus-quality chart.
def _record_cost(folder: Path, stage: str, seconds: float, usage: Dict[str, int]) -> None:
    costs = _read(folder / "stage_costs.json", {}) or {}
    costs[stage] = {"seconds": round(seconds, 2), **usage}
    _write(folder / "stage_costs.json", costs)


# Runs a pipeline stage only once and remembers the result as <name>.json.
#  - If the file exists and force is False, the saved result is returned (no LLM calls repeated).
#  - Otherwise it resets the token counters, runs the function, records the cost, saves and returns the result.
# This is why the long benchmark can be stopped (for example by the Groq daily quota) and resumed later.
def cached_stage(folder: Path, name: str, fn: Callable[[], Any], force: bool = False):
    path = folder / f"{name}.json"
    if path.exists() and not force:
        return _read(path)
    print(f"\n▶ [{folder.name}] stage: {name}")
    reset_token_usage()
    started = time.time()
    result = fn()
    _record_cost(folder, name, time.time() - started, snapshot_token_usage())
    _write(path, result)
    return result


# Imports app.py only when needed (it is heavy: FastAPI, LangGraph and the models),
# so the evaluation scripts start quickly.
def _app():
    import app  # imported lazily: heavy (FastAPI + LangGraph + models)

    return app


# Creates the shared ResearchState for one benchmark topic and one ablation mode,
# with the fixed benchmark years and any extra fields (papers, analysis, ...) the caller passes in.
def _state(topic: str, n: int, mode: str, **fields):
    app = _app()
    return app.ResearchState(
        topic=topic,
        start_year=START_YEAR,
        end_year=END_YEAR,
        max_papers=n,
        ablation_mode=mode,
        **fields,
    )


# ----------------------------------------------------------------------
# M0 - vanilla single-prompt baseline
# ----------------------------------------------------------------------

M0_PROMPT = """Write a rigorous academic literature review (Related Work section) on:

Research Topic:
{topic}

Cover approximately {n} representative peer-reviewed studies published between {start} and {end}.
Cite studies with numeric citations [1] ... [N] in the text.
The body (excluding References) should be 1000 to 1400 words.

OUTPUT FORMAT:
# Related Work
[Opening paragraph framing the research domain.]

## [Theme 1]
## [Theme 2]
## [Theme 3]
[2 to 4 thematic sections that synthesize and compare the studies.]

## Research Gaps and Open Directions
[Evidence-grounded discussion of limitations and open problems.]

## Conclusion

## References
[1] Authors, "Exact paper title," Venue, Year. doi: ...
(one entry per cited study, in numerical order)
"""


# Configuration M0 = vanilla baseline: ONE prompt, no retrieval, no agents.
# It asks the writer model to write a literature review about the topic directly from its own memory.
# This shows how much worse (more hallucination, fake references) a plain LLM is.
def run_m0(topic: str, n: int) -> Dict[str, Any]:
    app = _app()
    llm = make_llm(app.WRITER_MODEL, temperature=0.2, max_tokens=app.WRITER_MAX_TOKENS, reasoning_effort=app.WRITER_REASONING)
    review = invoke_text(llm, M0_PROMPT.format(topic=topic, n=n, start=START_YEAR, end=END_YEAR), label="M0 Vanilla")
    return {"final_review_md": review}


# ----------------------------------------------------------------------
# Pipeline configurations
# ----------------------------------------------------------------------

# Runs the last part of the pipeline for one configuration, exactly as in the LangGraph:
# summary adapter -> citation agent -> writing agent.
# It builds a state from the already computed papers, analysis, cross-paper result, gaps and taxonomy,
# and returns the review text, citations and verified gaps.
def _downstream(topic, papers, mode, analyzed, cross, gap_analysis, taxonomy):
    """summary adapter -> citations -> writer, exactly as in the LangGraph pipeline."""
    app = _app()
    gaps = (gap_analysis or {}).get("verified_gaps", []) or []
    state = _state(
        topic, len(papers), mode,
        raw_papers_list=papers,
        extracted_papers=analyzed,
        analyzed_papers=analyzed,
        cross_paper_analysis=cross or {},
        research_gap_analysis=gap_analysis or {},
        verified_research_gaps=gaps,
        taxonomy_map=taxonomy or {},
    )
    state.summarized_papers = app.summarization_agent_node(state)["summarized_papers"]
    state.citations_list = app.citation_agent_node(state)["citations_list"]
    written = app.writing_agent_node(state)
    return {
        "final_review_md": written["final_review_md"],
        "citations_list": state.citations_list,
        "verified_research_gaps": gaps,
    }


# Generates the reviews of the chosen configurations (M0..M5) for ONE topic.
#  - Builds the frozen benchmark papers first (same papers for every configuration).
#  - M0: single prompt. M1: abstracts only. M2: no cross-paper agent. M3: gaps without Skeptic/Verifier.
#    M4: full pipeline. M5: M4's draft plus the feedback loop.
#  - Expensive shared stages (analysis, taxonomy, cross-paper, gaps) are cached and reused, so the ONLY
#    difference between configurations is the ablated component.
def generate_topic(topic: str, configs: List[str], force: bool = False) -> None:
    import paper_analysis

    app = _app()
    paper_analysis.MAX_EVIDENCE_CHUNKS = ANALYSIS_MAX_CHUNKS
    paper_analysis.MAX_CHUNKS = ANALYSIS_MAX_CHUNKS

    folder = topic_dir(topic)
    build_topic(topic)
    papers = _read(folder / "papers.json")
    extracted = _read(folder / "extracted.json")
    n = len(papers)
    out = folder / "configs"

    needs_full = any(c in configs for c in ("M2_no_cross_paper", "M3_no_skeptic", "M4_full", "M5_full_feedback"))

    if "M0_vanilla" in configs:
        cached_stage(out / "M0_vanilla", "review", lambda: run_m0(topic, n), force)

    if "M1_retrieval_only" in configs:
        abstract_analyzed = app.abstract_only_analyses(papers)
        taxonomy_abs = cached_stage(
            folder, "taxonomy_abstract",
            lambda: app.taxonomy_agent_node(_state(topic, n, "M1_retrieval_only", analyzed_papers=abstract_analyzed))["taxonomy_map"],
            force,
        )
        cached_stage(
            out / "M1_retrieval_only", "review",
            lambda: _downstream(topic, papers, "M1_retrieval_only", abstract_analyzed, {}, {}, taxonomy_abs),
            force,
        )

    if not needs_full:
        return

    analyzed = cached_stage(folder, "analyzed", lambda: paper_analysis.analyze_papers(extracted), force)
    taxonomy = cached_stage(
        folder, "taxonomy_full",
        lambda: app.taxonomy_agent_node(_state(topic, n, "M4_full", analyzed_papers=analyzed))["taxonomy_map"],
        force,
    )

    if "M2_no_cross_paper" in configs:
        gaps_nocross = cached_stage(
            folder, "gaps_no_cross",
            lambda: app.research_gap_agent_node(
                _state(topic, n, "M2_no_cross_paper", analyzed_papers=analyzed, cross_paper_analysis={})
            )["research_gap_analysis"],
            force,
        )
        cached_stage(
            out / "M2_no_cross_paper", "review",
            lambda: _downstream(topic, papers, "M2_no_cross_paper", analyzed, {}, gaps_nocross, taxonomy),
            force,
        )

    if not any(c in configs for c in ("M3_no_skeptic", "M4_full", "M5_full_feedback")):
        return

    cross = cached_stage(folder, "cross", lambda: app.analyze_cross_papers(analyzed), force)
    gaps_full = cached_stage(
        folder, "gaps_full",
        lambda: app.research_gap_agent_node(
            _state(topic, n, "M4_full", analyzed_papers=analyzed, cross_paper_analysis=cross)
        )["research_gap_analysis"],
        force,
    )

    if "M3_no_skeptic" in configs:
        # Identical proposer candidates as M4, but without Skeptic/Verifier review.
        unverified = app.accept_candidates_without_verification(gaps_full.get("proposed_gaps", []))
        gaps_m3 = {
            "status": "UNVERIFIED_ABLATION",
            "proposed_gaps": gaps_full.get("proposed_gaps", []),
            "challenged_gaps": [],
            "verified_gaps": unverified,
        }
        cached_stage(
            out / "M3_no_skeptic", "review",
            lambda: _downstream(topic, papers, "M3_no_skeptic", analyzed, cross, gaps_m3, taxonomy),
            force,
        )

    if "M4_full" in configs or "M5_full_feedback" in configs:
        m4 = cached_stage(
            out / "M4_full", "review",
            lambda: _downstream(topic, papers, "M4_full", analyzed, cross, gaps_full, taxonomy),
            force,
        )

    if "M5_full_feedback" in configs:
        # Builds the M5 review: takes the finished M4 draft and its citations, runs the feedback agent
        # (critic -> fact-check -> reviser) on it, and returns the improved review, the old draft and the feedback log.
        def feedback():
            state = _state(
                topic, n, "M5_full_feedback",
                analyzed_papers=analyzed,
                citations_list=m4["citations_list"],
                final_review_md=m4["final_review_md"],
                feedback_rounds=app.FEEDBACK_MAX_ROUNDS,
            )
            result = app.feedback_agent_node(state)
            return {
                "final_review_md": result.get("final_review_md", m4["final_review_md"]),
                "draft_review_md": m4["final_review_md"],
                "feedback_log": result.get("feedback_log", []),
                "citations_list": m4["citations_list"],
            }

        cached_stage(out / "M5_full_feedback", "review", feedback, force)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

# Scores the generated reviews of ONE topic.
#  - For every configuration that has a review but no evaluation yet, evaluate_review() computes all metrics
#    against the topic's real papers and saves evaluation.json (and the review as review.md).
#  - Also judges the Skeptic/Verifier gap decisions (evaluate_gap_candidates) for the full and no-cross versions.
def evaluate_topic(topic: str, configs: List[str], force: bool = False, use_judge: bool = True) -> None:
    from .metrics import evaluate_gap_candidates, evaluate_review

    folder = topic_dir(topic)
    extracted = _read(folder / "extracted.json")
    if not extracted:
        return
    for config in configs:
        review_path = folder / "configs" / config / "review.json"
        eval_path = folder / "configs" / config / "evaluation.json"
        if not review_path.exists() or (eval_path.exists() and not force):
            continue
        review = (_read(review_path) or {}).get("final_review_md", "")
        print(f"\n🧮 Evaluating {folder.name} / {config}")
        started = time.time()
        reset_token_usage()
        result = evaluate_review(review, extracted, key=f"{folder.name}:{config}", use_judge=use_judge)
        result["metrics"]["eval_seconds"] = round(time.time() - started, 1)
        result["judge_usage"] = snapshot_token_usage()
        _write(eval_path, result)
        (folder / "configs" / config / "review.md").write_text(review, encoding="utf-8")

    gap_eval_path = folder / "gap_verification_eval.json"
    gaps_full = _read(folder / "gaps_full.json")
    analyzed = _read(folder / "analyzed.json")
    if gaps_full and analyzed and (force or not gap_eval_path.exists()):
        print(f"\n🧪 Evaluating Skeptic/Verifier gap decisions for {folder.name}")
        result = {"full": evaluate_gap_candidates(gaps_full, analyzed, extracted, use_judge)}
        gaps_nocross = _read(folder / "gaps_no_cross.json")
        if gaps_nocross:
            result["no_cross"] = evaluate_gap_candidates(gaps_nocross, analyzed, extracted, use_judge)
        _write(gap_eval_path, result)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

# Chooses which benchmark topics to run: the ones whose indexes were given on the command line, or all topics.
def _selected_topics(indices: Optional[List[int]]) -> List[str]:
    return [TOPICS[i] for i in indices] if indices else list(TOPICS)


# Command-line entry point:  python -m evaluation.ablation <command>
# Commands: build (get papers), generate (run M0..M5), evaluate (compute metrics), analyze (tables, statistics,
# charts), all (everything in order), status (what is finished).
# Options: --topics, --configs, --force, --no-judge. If the Groq quota is reached it stops cleanly and tells
# you to re-run later; finished stages are cached.
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MALRG ablation study runner")
    parser.add_argument("command", choices=["build", "generate", "evaluate", "analyze", "all", "status"])
    parser.add_argument("--topics", type=int, nargs="*", help="topic indices (default: all)")
    parser.add_argument("--configs", nargs="*", default=CONFIG_IDS, choices=CONFIG_IDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (NLI-only metrics)")
    args = parser.parse_args(argv)

    topics = _selected_topics(args.topics)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "status":
        from .analysis import print_status

        print_status(topics)
        return 0

    if args.command in {"build", "all"}:
        for topic in topics:
            try:
                print(build_topic(topic, force=args.force and args.command == "build"))
            except Exception as error:
                print(f"⚠️ benchmark build failed for '{topic}': {error}")

    halted = None
    if args.command in {"generate", "all"}:
        for topic in topics:
            try:
                generate_topic(topic, args.configs, force=args.force and args.command == "generate")
            except (DailyQuotaExhausted, LLMUnavailable) as error:
                halted = f"Groq quota/rate limit reached during '{topic}': {error}"
                print(f"\n🛑 {halted}\n   Re-run the same command later; finished stages are cached.")
                break
            except Exception:
                print(f"\n❌ generation failed for '{topic}':")
                traceback.print_exc()

    if args.command in {"evaluate", "all"}:
        for topic in topics:
            try:
                evaluate_topic(topic, args.configs, force=args.force and args.command == "evaluate", use_judge=not args.no_judge)
            except (DailyQuotaExhausted, LLMUnavailable) as error:
                print(f"\n🛑 judge quota reached during '{topic}': {error}. Re-run 'evaluate' later.")
                break
            except Exception:
                print(f"\n❌ evaluation failed for '{topic}':")
                traceback.print_exc()

    if args.command in {"analyze", "all"}:
        from .analysis import run_analysis

        run_analysis(topics)

    if halted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
