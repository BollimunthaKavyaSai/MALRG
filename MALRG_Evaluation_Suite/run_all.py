"""
End-to-end driver for the MALRG study.

    python run_all.py generate   # frozen benchmarks + all experiment configurations
    python run_all.py evaluate   # NLI metrics, LLM judge, retrieval + skeptic analysis
    python run_all.py analyze    # statistics, tables, figures, RESULTS.md, benchmark.json
    python run_all.py all        # everything (resumable: finished work is never redone)

User feedback (persists and is applied to every future review):
    python run_all.py feedback "Keep reviews under 900 words and report exact numbers."
    python run_all.py review "Graph neural networks for drug discovery"
    python run_all.py preferences          # show the learned preference rules

Options:
    --topics N            use the first N topics from config.TOPICS
    --groups a,b          experiment groups: ablation,baselines,prompts,preferences
    --no-judge            skip LLM-judge metrics (NLI metrics only)
"""

import argparse
import sys
import time
import traceback

import config
from malrg.llm import DailyLimitReached


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def generate(topics, configs):
    """Phase-major order (ablation first). A config whose model has no daily quota left is
    skipped; the remaining configs (other models) continue. Returns False if anything is pending."""
    from malrg.pipeline import run_config
    complete = True
    for c in configs:
        for i, topic in enumerate(topics, 1):
            if c in config.PROMPT_CONFIGS and topic not in config.PROMPT_TOPICS:
                continue
            try:
                r = run_config(topic, c, log=log)
                if r.get("_logged") is None:
                    u = r["usage"]
                    log(f"  [{c}] [{i}/{len(topics)}] done: {u['llm_calls']} calls, {u['total_tokens']} tokens, "
                        f"{len(r['review_md'].split())} words")
            except DailyLimitReached as exc:
                log(f"!! quota: {c} on topic {i} postponed ({str(exc)[:120]})")
                complete = False
            except Exception:
                log(f"!! {c} failed on '{topic}':\n{traceback.format_exc()}")
                complete = False
    return complete


def evaluate(topics, configs, use_judge=True):
    from evaluation import runner
    complete = True
    for i, topic in enumerate(topics, 1):
        log(f"=== evaluate [{i}/{len(topics)}] {topic}")
        try:
            if use_judge:
                runner.retrieval_eval(topic, log=log)
        except DailyLimitReached:
            complete = False
        except Exception:
            log(f"!! retrieval evaluation failed:\n{traceback.format_exc()}")
        for c in configs:
            if c in config.PROMPT_CONFIGS and topic not in config.PROMPT_TOPICS:
                continue
            try:
                if runner.evaluate_run(topic, c, log=log, use_judge=use_judge) is None:
                    complete = False
                if use_judge and c in ("M2", "M3", "M4"):
                    runner.skeptic_eval(topic, c, log=log)
            except DailyLimitReached:
                log(f"!! judge quota: evaluation of {c} postponed")
                complete = False
            except Exception:
                log(f"!! evaluation of {c} failed on '{topic}':\n{traceback.format_exc()}")
                complete = False
    return complete


def analyze(topics):
    from evaluation import report
    report.build(topics, list(config.ABLATION_CONFIGS), log=log)


def cmd_feedback(text):
    from malrg import preferences
    rules = preferences.record_feedback(text)
    print("Learned preferences (applied to every future review):")
    for r in rules:
        print(f"  - [{r.get('category', '')}] {r['rule']}")


def cmd_preferences():
    from malrg import preferences
    state = preferences.load()
    print(f"{len(state['feedback'])} feedback messages recorded.")
    for r in state["rules"]:
        print(f"  - [{r.get('category', '')}] {r['rule']}")


def cmd_review(topic):
    from malrg import preferences
    from malrg.pipeline import run_config
    cfg = {**config.CONFIGS[config.PROPOSED], "name": "User review (M5 + learned preferences)",
           "preferences": True, "preference_store": preferences.USER_STORE}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rec = run_config(topic, f"USER_{stamp}", log=log, cfg=cfg)
    path = config.RUNS_DIR / __import__("malrg.retrieval", fromlist=["slugify"]).slugify(topic)
    print(f"\nReview written to {path / f'USER_{stamp}_review.md'}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "evaluate", "analyze", "all", "feedback", "review",
                                      "preferences"])
    ap.add_argument("text", nargs="?", help="feedback text or review topic")
    ap.add_argument("--topics", type=int, default=len(config.TOPICS))
    ap.add_argument("--groups", default="ablation,baselines,prompts,preferences")
    ap.add_argument("--configs", default=None, help="explicit config ids, overrides --groups")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    if args.stage == "feedback":
        cmd_feedback(args.text)
        return 0
    if args.stage == "preferences":
        cmd_preferences()
        return 0
    if args.stage == "review":
        cmd_review(args.text)
        return 0

    topics = config.TOPICS[:args.topics]
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        configs = [c for g in args.groups.split(",") for c in config.GROUPS[g.strip()]]

    ok = True
    if args.stage in ("generate", "all"):
        ok = generate(topics, configs)
    if args.stage in ("evaluate", "all"):
        # evaluate whatever is already generated: the judge model has its own quota
        ok = evaluate(topics, configs, use_judge=not args.no_judge) and ok
    if args.stage in ("analyze", "all"):
        try:
            analyze(topics)
        except Exception:
            log(f"!! analysis failed:\n{traceback.format_exc()}")
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
