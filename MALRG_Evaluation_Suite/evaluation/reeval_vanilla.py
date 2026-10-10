"""
Re-score the no-retrieval configurations (M0, B1, B2) with the corrected reference checker
(arXiv + Crossref; lookup failures are 'unverified' and never counted as fabricated).
Recomputes the NLI metrics, sentence rows and extended metrics; LLM-judge scores are kept.

    python -m evaluation.reeval_vanilla
"""

import json
import time

import numpy as np

import config
from evaluation import metrics
from evaluation.extra_metrics import extended_metrics
from evaluation.nli import get_verifier
from malrg.retrieval import benchmark_path, slugify

VANILLA = [c for c, v in config.CONFIGS.items() if not v.get("retrieval", True)]


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def main():
    verifier = get_verifier("eval")
    for t in config.TOPICS:
        d = config.RUNS_DIR / slugify(t)
        bench = json.loads(benchmark_path(t).read_text(encoding="utf-8"))
        for c in VANILLA:
            ev_p = d / f"{c}_eval.json"
            run = json.loads((d / f"{c}.json").read_text(encoding="utf-8"))
            ev = json.loads(ev_p.read_text(encoding="utf-8"))
            if ev.get("reference_checker") == "arxiv+crossref-v2":
                continue
            t0 = time.time()
            m, rows = metrics.evaluate_review(run, bench, verifier)
            ev["nli"] = m
            ev["sentences"] = rows
            ev["extended"] = extended_metrics(run, bench, rows)
            ev["reference_checker"] = "arxiv+crossref-v2"
            ev_p.write_text(json.dumps(ev, ensure_ascii=False, indent=1, default=_default), encoding="utf-8")
            print(f"{time.strftime('%H:%M:%S')} {c} {t[:40]:40} refs={m['n_references']} "
                  f"unverifiable={m.get('n_unverifiable_references')} FRR={m['fabricated_reference_rate']:.2f} "
                  f"CG={m['citation_groundedness']:.2f} HR={m['hallucination_rate']:.2f} ({time.time() - t0:.0f}s)",
                  flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
