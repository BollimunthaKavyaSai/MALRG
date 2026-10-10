"""
Recompute the Fabricated Reference Rate for the no-retrieval configurations after the
truncation fix (a review cut off by the output-length limit has no judgeable reference list,
and an incomplete last reference is excluded). Only the FRR fields are updated.

    python -m evaluation.recompute_frr
"""

import json

import numpy as np

import config
from evaluation.metrics import vanilla_sources
from malrg.retrieval import benchmark_path, slugify

VANILLA = [c for c, v in config.CONFIGS.items() if not v.get("retrieval", True)]


def main():
    rows = []
    for t in config.TOPICS:
        d = config.RUNS_DIR / slugify(t)
        bench = json.loads(benchmark_path(t).read_text(encoding="utf-8"))
        for c in VANILLA:
            run_p, ev_p = d / f"{c}.json", d / f"{c}_eval.json"
            if not (run_p.exists() and ev_p.exists()):
                continue
            run = json.loads(run_p.read_text(encoding="utf-8"))
            ev = json.loads(ev_p.read_text(encoding="utf-8"))
            refs, pseudo, fabricated, _ = vanilla_sources(run["review_md"], bench)
            n = len(refs)
            frr = len(fabricated) / n if n else float("nan")
            old = ev["nli"].get("fabricated_reference_rate")
            ev["nli"]["fabricated_reference_rate"] = frr
            ev["nli"]["n_references"] = n
            ev["nli"]["n_fabricated_references"] = len(fabricated)
            ev_p.write_text(json.dumps(ev, ensure_ascii=False, indent=1,
                                       default=lambda o: float(o) if isinstance(o, np.floating) else o),
                            encoding="utf-8")
            rows.append((c, t[:40], n, len(fabricated), old, frr))
            print(f"{c} {t[:45]:45} refs={n:2d} fabricated={len(fabricated):2d}  FRR {old:.2f} -> {frr:.2f}")
    for c in VANILLA:
        vals = [r[5] for r in rows if r[0] == c and not np.isnan(r[5])]
        tot_refs = sum(r[2] for r in rows if r[0] == c)
        tot_fab = sum(r[3] for r in rows if r[0] == c)
        print(f"{c}: mean FRR over {len(vals)} topics with references = {np.mean(vals):.3f}; "
              f"pooled {tot_fab}/{tot_refs} = {tot_fab / max(1, tot_refs):.3f}")


if __name__ == "__main__":
    main()
