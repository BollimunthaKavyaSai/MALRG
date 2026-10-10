"""Evaluates every logged run and every topic; results are cached next to the runs."""

import json
import math

import numpy as np
import pandas as pd

import config
from evaluation import judge, metrics
from evaluation.nli import get_verifier
from malrg.retrieval import RANKERS, benchmark_path, ranking, slugify


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_np_default), encoding="utf-8")


def _np_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def evaluate_run(topic, config_id, log=print, use_judge=True):
    d = config.RUNS_DIR / slugify(topic)
    run = _load(d / f"{config_id}.json")
    if run is None:
        return None
    out_path = d / f"{config_id}_eval.json"
    cached = _load(out_path)
    if cached and (cached.get("judge_done") or not use_judge):
        return cached
    bench = _load(benchmark_path(topic))
    res = dict(cached or {})
    if "nli" not in res:
        m, rows = metrics.evaluate_review(run, bench, get_verifier("eval"))
        res["nli"] = m
        res["sentences"] = rows
        res["extraction"] = metrics.extraction_metrics(run)
        res["gap_pipeline"] = metrics.gap_pipeline_metrics(run)
        res["efficiency"] = metrics.efficiency_metrics(run)
        from malrg.preferences import check_compliance
        res["compliance"] = check_compliance(run["review_md"])
        log(f"    [eval {config_id}] CG={m['citation_groundedness']:.2f} CP={m['citation_precision']:.2f} "
            f"HR={m['hallucination_rate']:.2f} SC={m['source_coverage']:.2f} MSR={m['multi_source_synthesis']:.2f}")
    if use_judge and not res.get("judge_done"):
        scores, raw = judge.rubric_judge(run, bench)
        res["judge"] = scores
        res["judge_raw"] = raw
        res["judge_claims"] = (judge.claim_judge(run, bench, res["sentences"])
                               if config_id in config.CLAIM_JUDGE_CONFIGS else {})
        res["judge_done"] = True
        log(f"    [judge {config_id}] overall={scores.get('judge_overall')} gap_precision="
            f"{scores.get('judge_gap_precision'):.2f}")
    _dump(out_path, res)
    return res


def retrieval_eval(topic, log=print):
    d = config.RUNS_DIR / slugify(topic)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "retrieval_eval.json"
    cached = _load(path)
    if cached:
        return cached
    bench = _load(benchmark_path(topic))
    if bench is None:          # topic not generated yet
        return None
    cands = bench["candidates"]
    tops = {rk: ranking(cands, rk)[:10] for rk in RANKERS}
    pool, seen = [], {}
    for rk in RANKERS:
        for c in tops[rk]:
            if c["title"] not in seen:
                seen[c["title"]] = len(pool)
                pool.append(c)
    for p in bench["papers"]:
        if p["title"] not in seen:
            seen[p["title"]] = len(pool)
            pool.append(p)
    grades = judge.relevance_judge(bench, pool)
    g_of = {t: grades.get(i, 0) for t, i in seen.items()}
    rows = []
    for rk in RANKERS:
        rel = [g_of[c["title"]] for c in tops[rk]]
        rows.append({"topic": topic, "ranker": rk,
                     "P@5": float(np.mean([r >= 1 for r in rel[:5]])),
                     "P@10": float(np.mean([r >= 1 for r in rel[:10]])),
                     "nDCG@10": _ndcg(rel, sorted(g_of.values(), reverse=True)),
                     "MRR": next((1 / (i + 1) for i, r in enumerate(rel) if r == 2), 0.0)})
    selected = [g_of[p["title"]] for p in bench["papers"]]
    res = {"rows": rows, "selected_mean_grade": float(np.mean(selected)),
           "selected_relevant_rate": float(np.mean([g >= 1 for g in selected])), "pool_size": len(pool)}
    _dump(path, res)
    log(f"    [retrieval] hybrid nDCG@10={rows[0]['nDCG@10']:.2f}  selected relevant="
        f"{res['selected_relevant_rate']:.2f}")
    return res


def _ndcg(rel, ideal_pool, k=10):
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_pool[:k]))
    return dcg / idcg if idcg else 0.0


def skeptic_eval(topic, config_id, log=print):
    """Silver-label analysis of the gap filter (Skeptic precision etc.)."""
    d = config.RUNS_DIR / slugify(topic)
    path = d / f"{config_id}_gapjudge.json"
    cached = _load(path)
    if cached is not None:
        return cached
    run = _load(d / f"{config_id}.json")
    if not run or not run.get("gap_result"):
        return None
    bench = _load(benchmark_path(topic))
    g = run["gap_result"]
    proposed = g.get("proposed", [])
    labels = judge.candidate_judge(bench, proposed)
    if g.get("mode") == "adversarial_triad":
        gated = g.get("gated", [])
        verified_idx = set()
        for v in g.get("final", []):
            try:
                verified_idx.add(int(str(v.get("id", "G-1"))[1:]))
            except ValueError:
                continue
        accepted_texts = {gated[i]["gap"] for i in verified_idx if 0 <= i < len(gated)}
    else:
        accepted_texts = {p["gap"] for p in proposed}
    acc, rej = [], []
    for i, p in enumerate(proposed):
        if i not in labels:
            continue
        (acc if p["gap"] in accepted_texts else rej).append(labels[i])
    res = {
        "n_proposed": len(proposed),
        "candidate_valid_rate": float(np.mean(list(labels.values()))) if labels else np.nan,
        "acceptance_precision": float(np.mean(acc)) if acc else np.nan,
        "skeptic_precision": float(np.mean([not x for x in rej])) if rej else np.nan,
        "n_accepted": len(acc), "n_rejected": len(rej),
    }
    _dump(path, res)
    return res


def collect_table(topics, configs):
    rows = []
    for t in topics:
        d = config.RUNS_DIR / slugify(t)
        for c in configs:
            ev = _load(d / f"{c}_eval.json")
            if not ev:
                continue
            row = {"topic": t, "config_id": c}
            row.update(ev.get("nli", {}))
            row.update(ev.get("judge", {}))
            row.update(ev.get("judge_claims", {}))
            row.update({f"extr_{k}": v for k, v in (ev.get("extraction") or {}).items()})
            row.update({f"gp_{k}": v for k, v in (ev.get("gap_pipeline") or {}).items()})
            row.update(ev.get("efficiency", {}))
            if "compliance" not in ev and (d / f"{c}.json").exists():
                from malrg.preferences import check_compliance
                ev["compliance"] = check_compliance(_load(d / f"{c}.json")["review_md"])
            row.update({f"pref_{k}": v for k, v in (ev.get("compliance") or {}).items()})
            if "extended" not in ev and ev.get("sentences") is not None:
                from evaluation.extra_metrics import extended_metrics
                run = _load(d / f"{c}.json")
                bench = _load(benchmark_path(t))
                if run and bench:
                    ev["extended"] = extended_metrics(run, bench, ev["sentences"])
                    _dump(d / f"{c}_eval.json", ev)   # cache: computed once per run
            row.update(ev.get("extended") or {})
            gj = _load(d / f"{c}_gapjudge.json")
            if gj:
                row.update({f"gj_{k}": v for k, v in gj.items()})
            rows.append(row)
    return pd.DataFrame(rows)
