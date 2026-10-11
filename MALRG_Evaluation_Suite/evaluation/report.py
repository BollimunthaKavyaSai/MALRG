"""Builds all tables, figures and the results report from the evaluated runs."""

import json
import re
import time

import numpy as np
import pandas as pd

import config
from evaluation import figures, stats
from evaluation.runner import collect_table
from malrg.retrieval import slugify

CORE = ["citation_groundedness", "citation_precision", "hallucination_rate", "fabricated_reference_rate",
        "uncited_claim_rate", "source_coverage", "multi_source_synthesis", "cross_paper_comparison",
        "judge_claim_support", "judge_gap_precision", "judge_synthesis", "judge_critical_analysis",
        "judge_overall", "overall_quality_index"]
PPT = ["source_coverage", "synthesis_indicators_per_1k", "gap_indicators", "structure_score",
       "certainty_flags", "redundancy", "word_count"]
EFF = ["llm_calls", "total_tokens"]
EXTENDED = ["soft_groundedness", "numeric_fidelity", "semantic_alignment", "source_content_recall",
            "topic_relevance", "citation_density", "lexical_diversity", "readability_fk_grade",
            "flesch_reading_ease"]


def _cell(sumdf, c, m, best=None):
    r = sumdf[sumdf.config_id == c]
    if r.empty or f"{m}_mean" not in r:
        return "n/a"
    mu = r[f"{m}_mean"].iloc[0]
    sd = r[f"{m}_std"].iloc[0]
    if np.isnan(mu):
        return "n/a"
    pct = stats.METRICS[m][2]
    s = (f"{100 * mu:.1f} ± {100 * sd:.1f}" if pct else
         f"{mu:.0f} ± {sd:.0f}" if m in ("word_count", "total_tokens") else f"{mu:.2f} ± {sd:.2f}")
    return f"**{s}**" if best == c else s


def _best(sumdf, m):
    higher = stats.METRICS[m][1]
    if higher is None or f"{m}_mean" not in sumdf:
        return None
    s = sumdf.set_index("config_id")[f"{m}_mean"].dropna()
    if s.empty:
        return None
    return s.idxmax() if higher else s.idxmin()


def _safe(fn, *args, **kwargs):
    """A figure that cannot be drawn from partial data must never block the report."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"  [figure skipped] {fn.__name__}: {exc}")
        return None


def markdown_table(sumdf, metrics, configs):
    head = "| Configuration | " + " | ".join(
        f"{stats.METRICS[m][0]} {'↑' if stats.METRICS[m][1] else '↓' if stats.METRICS[m][1] is False else ''}"
        for m in metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines = [head, sep]
    for c in configs:
        lines.append(f"| {config.CONFIGS[c]['name']} | " +
                     " | ".join(_cell(sumdf, c, m, _best(sumdf, m)) for m in metrics) + " |")
    return "\n".join(lines)


def latex_table(sumdf, metrics, configs, caption, label):
    cols = "l" + "c" * len(metrics)
    up, down = "$\\uparrow$", "$\\downarrow$"

    def arrow(m):
        return up if stats.METRICS[m][1] else down if stats.METRICS[m][1] is False else ""

    head = " & ".join(["\\textbf{Configuration}"] + [
        "\\textbf{" + stats.METRICS[m][0] + "} " + arrow(m) for m in metrics]) + " \\\\"
    body = []
    for c in configs:
        cells = []
        for m in metrics:
            r = sumdf[sumdf.config_id == c]
            mu = r[f"{m}_mean"].iloc[0] if not r.empty and f"{m}_mean" in r else np.nan
            v = "n/a" if np.isnan(mu) else (f"{100 * mu:.1f}" if stats.METRICS[m][2] else f"{mu:.2f}")
            cells.append(f"\\textbf{{{v}}}" if _best(sumdf, m) == c else v)
        name = config.CONFIGS[c]["name"].replace("&", "\\&")
        body.append(" & ".join([name] + cells) + " \\\\")
    return ("\\begin{table*}[t]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")


def significance_md(tests):
    if tests.empty:
        return "_not enough paired topics_"
    lines = ["| Metric | Proposed vs. | Δ (abs) | Δ (rel) | Wilcoxon p | Holm p | Cohen's dz | r_rb | Win rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        pct = stats.METRICS[r.metric][2]
        d = f"{100 * r.delta:+.1f} pp" if pct else f"{r.delta:+.2f}"
        rel = "n/a" if np.isnan(r.rel_delta_pct) else f"{r.rel_delta_pct:+.0f}%"
        sig = " ✓" if r.significant else ""
        dz = "∞" if np.isinf(r.cohens_dz) else f"{r.cohens_dz:.2f}"
        wr = "n/a" if np.isnan(r.win_rate) else f"{100 * r.win_rate:.0f}%"
        lines.append(f"| {stats.METRICS[r.metric][0]} | {r.baseline} | {d} | {rel} | {r.wilcoxon_p:.4f} | "
                     f"{r.p_holm:.4f}{sig} | {dz} | {r.rank_biserial:.2f} | {wr} |")
    return "\n".join(lines)


def build(topics, configs, log=print):
    df = collect_table(topics, configs)
    if df.empty:
        log("no evaluated runs yet")
        return
    for col in ["judge_gap_precision", "judge_overall", "judge_claim_support"]:
        if col not in df:
            df[col] = np.nan
    df = stats.add_quality_index(df)
    # keep only topics that have every configuration (strictly paired design)
    complete = df.groupby("topic")["config_id"].nunique()
    full_topics = complete[complete == len(configs)].index
    dfp = df[df.topic.isin(full_topics)].copy()
    log(f"topics with all {len(configs)} configs: {len(full_topics)} / {df.topic.nunique()}")
    if dfp.empty:
        dfp = df

    T = config.TABLES_DIR
    dfp.to_csv(T / "per_run_metrics.csv", index=False)
    sumdf = stats.summary_table(dfp)
    sumdf.to_csv(T / "ablation_summary.csv", index=False)

    test_cols = ["metric", "baseline", "n_topics", "proposed_mean", "baseline_mean", "delta", "rel_delta_pct",
                 "wilcoxon_p", "cohens_dz", "rank_biserial", "win_rate", "p_holm", "significant"]
    parts = [t for t in (stats.paired_tests(dfp, m) for m in CORE + PPT[1:] + EFF if m in dfp) if not t.empty]
    # with < 2 complete topics no paired test is possible yet: keep an empty, well-formed table
    tests = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=test_cols)
    tests.to_csv(T / "significance_tests.csv", index=False)
    fried = pd.DataFrame([f for f in (stats.friedman(dfp, m) for m in CORE if m in dfp) if f])
    fried.to_csv(T / "friedman_tests.csv", index=False)
    contrib = stats.component_contributions(dfp)
    contrib.to_csv(T / "component_contributions.csv", index=False)

    # retrieval
    ret_rows, sel = [], []
    for t in full_topics:
        p = config.RUNS_DIR / slugify(t) / "retrieval_eval.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            ret_rows += r["rows"]
            sel.append(r["selected_relevant_rate"])
    ret_df = pd.DataFrame(ret_rows)
    if not ret_df.empty:
        ret_df.to_csv(T / "retrieval_ablation.csv", index=False)

    # figures
    F = {}
    F["ground"] = _safe(figures.ablation_bars, dfp, ["citation_groundedness", "citation_precision",
                                              "hallucination_rate", "fabricated_reference_rate"],
                                       "fig1_grounding.png", "Grounding & hallucination")
    F["synth"] = _safe(figures.ablation_bars, dfp, ["source_coverage", "multi_source_synthesis",
                                             "cross_paper_comparison", "judge_gap_precision"],
                                      "fig2_synthesis.png", "Coverage, synthesis & research gaps")
    F["judge"] = _safe(figures.ablation_bars, dfp, ["judge_synthesis", "judge_critical_analysis",
                                             "judge_gap_quality", "judge_overall"],
                                      "fig3_judge.png", "LLM-judge quality (blind, 1-10)")
    F["radar"] = _safe(figures.radar, dfp, ["citation_groundedness", "citation_precision", "hallucination_rate",
                                     "source_coverage", "multi_source_synthesis", "judge_gap_precision",
                                     "judge_overall", "structure_score"], "fig4_radar.png")
    F["contrib"] = _safe(figures.contributions, contrib, "fig5_contributions.png")
    runs = []
    for t in full_topics:
        p = config.RUNS_DIR / slugify(t) / f"{config.PROPOSED}.json"
        if p.exists():
            runs.append(json.loads(p.read_text(encoding="utf-8")))
    F["feedback"] = _safe(figures.feedback_curve, runs, "fig6_feedback_loop.png")
    F["heat"] = _safe(figures.topic_heatmap, dfp, "citation_groundedness", "fig7_topic_heatmap.png")
    F["box"] = _safe(figures.distribution, dfp, "hallucination_rate", "fig8_hallucination_box.png")
    F["ret"] = _safe(figures.retrieval_bars, ret_df, "fig9_retrieval.png")
    F["cost"] = _safe(figures.cost_quality, dfp, "fig10_cost_quality.png")

    cfgs = [c for c in configs if c in set(dfp.config_id)]
    md_main = markdown_table(sumdf, ["citation_groundedness", "citation_precision", "hallucination_rate",
                                     "fabricated_reference_rate", "source_coverage", "multi_source_synthesis",
                                     "judge_gap_precision", "judge_overall", "overall_quality_index"], cfgs)
    md_ppt = markdown_table(sumdf, PPT, cfgs)
    md_judge = markdown_table(sumdf, ["judge_coverage", "judge_synthesis", "judge_critical_analysis",
                                      "judge_gap_quality", "judge_coherence", "judge_faithfulness",
                                      "judge_claim_support"], cfgs)
    md_eff = markdown_table(sumdf, ["llm_calls", "total_tokens", "word_count"], cfgs)
    md_ext = markdown_table(sumdf, [m for m in EXTENDED if f"{m}_mean" in sumdf], cfgs)
    (T / "ablation_main.tex").write_text(latex_table(
        sumdf, ["citation_groundedness", "citation_precision", "hallucination_rate",
                "fabricated_reference_rate", "source_coverage", "multi_source_synthesis",
                "judge_gap_precision", "judge_overall", "overall_quality_index"], cfgs,
        f"Ablation study of MALRG components (mean over {len(full_topics)} topics; best in bold).",
        "tab:ablation"), encoding="utf-8")

    sig_main = tests[tests.metric.isin(["citation_groundedness", "citation_precision", "hallucination_rate",
                                        "source_coverage", "multi_source_synthesis", "judge_gap_precision",
                                        "judge_overall", "overall_quality_index"])]
    fried_md = "\n".join(
        f"| {stats.METRICS[r.metric][0]} | {r.chi2:.2f} | {r.p:.2e} | {r.kendall_w:.2f} |"
        for _, r in fried.iterrows()) if not fried.empty else ""

    contrib_md = ["| Component | Δ GQI | Δ Citation Groundedness | Δ Hallucination Rate | Δ Multi-Source Synthesis | Δ Gap Precision | Δ Judge Overall |",
                  "|---|---|---|---|---|---|---|"]
    for comp in contrib.component.unique() if not contrib.empty else []:
        sub = contrib[contrib.component == comp].set_index("metric")

        def g(m, pct=True):
            if m not in sub.index:
                return "n/a"
            d, p = sub.loc[m, "delta"], sub.loc[m, "p"]
            s = f"{100 * d:+.1f} pp" if pct else f"{d:+.2f}"
            return s + ("*" if p < 0.05 else "")
        contrib_md.append(f"| {comp} | {g('overall_quality_index')} | {g('citation_groundedness')} | "
                          f"{g('hallucination_rate')} | {g('multi_source_synthesis')} | "
                          f"{g('judge_gap_precision')} | {g('judge_overall', False)} |")

    ret_md = ""
    if not ret_df.empty:
        agg = ret_df.groupby("ranker")[["P@5", "P@10", "nDCG@10", "MRR"]].mean()
        ret_md = "| Ranker | P@5 | P@10 | nDCG@10 | MRR |\n|---|---|---|---|---|\n" + "\n".join(
            f"| {rk} | {agg.loc[rk, 'P@5']:.3f} | {agg.loc[rk, 'P@10']:.3f} | {agg.loc[rk, 'nDCG@10']:.3f} | "
            f"{agg.loc[rk, 'MRR']:.3f} |" for rk in ["hybrid", "semantic", "tfidf", "title"] if rk in agg.index)
        ret_md += f"\n\nFrozen source sets: {100 * np.mean(sel):.1f}% of selected papers judged relevant (grade ≥ 1)."

    gap_md = ""
    gcols = [c for c in dfp.columns if c.startswith("gj_")]
    if gcols:
        gsub = dfp[dfp.config_id.isin(["M2", "M3", "M4"])].groupby("config_id")[
            ["gj_candidate_valid_rate", "gj_acceptance_precision", "gj_skeptic_precision",
             "gj_n_accepted", "gj_n_rejected"]].mean()
        gap_md = ("| Config | Candidate validity | Acceptance precision | Skeptic (rejection) precision | "
                  "Accepted / topic | Rejected / topic |\n|---|---|---|---|---|---|\n")
        for c, r in gsub.iterrows():
            gap_md += (f"| {c} | {100 * r.gj_candidate_valid_rate:.1f}% | {100 * r.gj_acceptance_precision:.1f}% | "
                       f"{'n/a' if np.isnan(r.gj_skeptic_precision) else f'{100 * r.gj_skeptic_precision:.1f}%'} | "
                       f"{r.gj_n_accepted:.1f} | {r.gj_n_rejected:.1f} |\n")

    extr_md = ""
    if "extr_evidence_fidelity" in dfp:
        e = dfp[dfp.config_id == "M4"][["extr_evidence_fidelity", "extr_field_completeness"]].mean()
        extr_md = (f"Paper Analysis Agent (M2-M5): **{100 * e.extr_evidence_fidelity:.1f}%** of extracted evidence "
                   f"quotes were found verbatim in the source PDF (Evidence Gate pass rate); field completeness "
                   f"**{100 * e.extr_field_completeness:.1f}%**.")

    fb_md = ""
    if runs:
        rounds = []
        for r in runs:
            for h in r.get("feedback_history") or []:
                if isinstance(h.get("round"), int):
                    rounds.append(h)
        if rounds:
            tot = lambda k: sum(h.get(k, 0) or 0 for h in rounds)
            fb_md = (f"Across {len(runs)} topics the feedback loop performed {tot('deterministic_recitations')} "
                     f"deterministic re-citations, {tot('llm_accepted')} verified LLM rewrites, "
                     f"{tot('llm_deleted')} deletions of unverifiable sentences, rejected "
                     f"{tot('llm_rejected_by_guard')} rewrites that failed verification (kept original), and added "
                     f"{tot('coverage_additions')} coverage sentences for uncited sources.")

    results = {
        "n_topics": int(len(full_topics)), "configs": cfgs, "main_table": md_main, "ppt_table": md_ppt,
        "judge_table": md_judge, "eff_table": md_eff, "ext_table": md_ext, "significance": significance_md(sig_main),
        "friedman": fried_md, "contrib": "\n".join(contrib_md), "retrieval": ret_md, "gap_table": gap_md,
        "extraction": extr_md, "feedback": fb_md, "figures": {k: (v.name if v else None) for k, v in F.items()},
    }
    (config.RESULTS_DIR / "results_summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    benchmark_json(sumdf, cfgs, len(full_topics), tests=tests, contrib=contrib,
                   gap_df=dfp[[c for c in dfp.columns if c.startswith('gj_')] + ['config_id']]
                   if any(c.startswith('gj_') for c in dfp.columns) else None)
    from evaluation.experiments_report import build_all
    results["experiments"] = build_all(topics, log=log)
    (config.RESULTS_DIR / "results_summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    results_md = results_markdown(results)
    (config.RESULTS_DIR / "RESULTS.md").write_text(results_md, encoding="utf-8")
    # full report = methodology + results (figure paths relative to the suite root)
    _meth = config.ROOT / "REPORT_METHODOLOGY.md"
    method = _meth.read_text(encoding="utf-8") if _meth.exists() else "# MALRG evaluation report\n"
    body = results_md.replace("](figures/", "](results/figures/")
    body = re.sub(r"^## (\d+)\. ", lambda m: f"### 7.{m.group(1)} ", body, flags=re.M)
    body = body.replace("# MALRG Ablation Study — Results", "## 7. Results", 1)
    status = (f"\n> Status: generated {time.strftime('%Y-%m-%d %H:%M')} from {len(full_topics)} complete "
              f"topics (target {len(topics)}). Re-run `python run_all.py analyze` to refresh.\n")
    (config.ROOT / "REPORT.md").write_text(method + "\n\n" + body.replace("## 7. Results", "## 7. Results" + status, 1),
                                           encoding="utf-8")
    log(f"wrote tables -> {T}, figures -> {config.FIGURES_DIR}, results -> RESULTS.md")


def benchmark_json(sumdf, cfgs, n_topics, tests=None, contrib=None, gap_df=None):
    """Machine-readable results consumed by the web app's Ablation Benchmark tab
    (schema of frontend/src/components/AblationDashboard.jsx)."""
    import time
    primary = ["overall_quality_index", "citation_groundedness", "citation_precision", "hallucination_rate",
               "fabricated_reference_rate", "source_coverage", "multi_source_synthesis",
               "judge_gap_precision", "judge_overall"]
    secondary = ["cross_paper_comparison", "uncited_claim_rate", "judge_claim_support", "judge_synthesis",
                 "judge_critical_analysis", "judge_gap_quality", "judge_faithfulness", "structure_score",
                 "synthesis_indicators_per_1k", "gap_indicators", "certainty_flags", "redundancy",
                 "word_count", "total_tokens", "llm_calls"]

    def kind(m):
        return "rate" if stats.METRICS[m][2] else ("score" if m.startswith("judge_") or m in (
            "citation_balance", "synthesis_indicators_per_1k") else "count")

    def num(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

    metrics_meta = [{"key": m, "label": stats.METRICS[m][0], "kind": kind(m),
                     "higher_is_better": bool(stats.METRICS[m][1]), "primary": m in primary}
                    for m in primary + secondary if f"{m}_mean" in sumdf]
    table = {}
    for c in cfgs:
        r = sumdf[sumdf.config_id == c].iloc[0]
        table[c] = {m["key"]: {"mean": num(r.get(f"{m['key']}_mean")), "std": num(r.get(f"{m['key']}_std")),
                               "ci_low": num(r.get(f"{m['key']}_ci_lo")), "ci_high": num(r.get(f"{m['key']}_ci_hi"))}
                    for m in metrics_meta}
    test_rows = []
    if tests is not None and not tests.empty:
        for _, t in tests.iterrows():
            test_rows.append({"metric": t.metric, "a": config.PROPOSED, "b": t.baseline,
                              "delta": num(t.delta), "p": num(t.wilcoxon_p), "p_holm": num(t.p_holm),
                              "effect_dz": None if np.isinf(t.cohens_dz) else num(t.cohens_dz)})
    contribs = []
    if contrib is not None and not contrib.empty:
        for comp in contrib.component.unique():
            sub = contrib[contrib.component == comp].set_index("metric")["delta"]
            contribs.append({"component": comp, **{m: num(sub.get(m)) for m in primary}})
    gv = {}
    if gap_df is not None and not gap_df.empty:
        g = gap_df.groupby("config_id").mean(numeric_only=True)
        gv = {"precision_unfiltered_M3": num(g.loc["M3", "gj_acceptance_precision"]) if "M3" in g.index else None,
              "precision_verified_M4": num(g.loc["M4", "gj_acceptance_precision"]) if "M4" in g.index else None,
              "skeptic_precision": num(g.loc["M4", "gj_skeptic_precision"]) if "M4" in g.index else None,
              "rejection_rate": num(g.loc["M4", "gj_n_rejected"] / max(1e-9, g.loc["M4", "gj_n_rejected"]
                                                                      + g.loc["M4", "gj_n_accepted"]))
              if "M4" in g.index else None}
    charts = sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))
    out = {"available": True, "source": "MALRG_Evaluation_Suite",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_topics": n_topics,
           "papers_per_topic": config.PAPERS_PER_TOPIC, "judge_model": config.MODELS["judge"],
           "proposed": config.PROPOSED, "headline": primary[:6],
           "configs": [{"id": c, "label": config.CONFIGS[c]["name"]} for c in cfgs],
           "metrics": metrics_meta, "table": table, "tests": test_rows, "contributions": contribs,
           "gap_verification": {"mean": gv}, "charts": charts,
           "latex_table": (config.TABLES_DIR / "ablation_main.tex").read_text(encoding="utf-8")}
    (config.RESULTS_DIR / "benchmark.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def results_markdown(r):
    fig = lambda k, cap: f"\n![{cap}](figures/{r['figures'][k]})\n" if r["figures"].get(k) else ""
    return f"""# MALRG Ablation Study — Results

All numbers below are computed automatically by `run_all.py` from the logged runs
(mean ± std over **{r['n_topics']} topics**, {config.PAPERS_PER_TOPIC} frozen full-text papers per topic).
Best value per column in bold.

## 1. Main ablation table
{r['main_table']}
{fig('ground', 'Grounding')}{fig('synth', 'Synthesis')}
## 2. Statistical significance (proposed {config.PROPOSED} vs. each configuration)
Paired Wilcoxon signed-rank over topics, Holm-Bonferroni corrected (✓ = significant at α = 0.05).

{r['significance']}

Friedman omnibus test across all configurations:

| Metric | χ² | p | Kendall's W |
|---|---|---|---|
{r['friedman']}

## 3. Component contribution (paired Δ, * p < 0.05)
{r['contrib']}
{fig('contrib', 'Contributions')}{fig('radar', 'Radar')}
## 4. LLM-judge quality (blind, 1–10)
{r['judge_table']}
{fig('judge', 'Judge')}
## 5. Review-1 (slide 28) indicators
{r['ppt_table']}

## 6. Research-gap verification (silver labels)
{r['gap_table']}

## 7. Feedback loop
{r['feedback']}
{fig('feedback', 'Feedback loop')}
## 8. Retrieval ablation (silver relevance labels)
{r['retrieval']}
{fig('ret', 'Retrieval')}
## 9. Evidence extraction
{r['extraction']}

## 10. Efficiency
{r['eff_table']}
{fig('cost', 'Cost vs quality')}{fig('heat', 'Per-topic heatmap')}{fig('box', 'Hallucination distribution')}
{_experiment_sections(r.get('experiments') or {})}
## 15. Extended evaluation metrics (ablation)
Soft groundedness (mean entailment), numeric fidelity (reported numbers found in the cited source text),
semantic source alignment, source content recall, topic relevance, citation density, lexical diversity
(MTLD) and readability. All computed locally, without LLM calls or human labels.

{r.get('ext_table', '')}
"""


def _experiment_sections(ex):
    titles = {"baselines": "11. Baseline models (vanilla LLMs and RAG) vs. proposed system",
              "prompts": "12. Prompt comparison (5 writer prompts, identical evidence)",
              "preferences": "13. User-feedback learning (applied next time)",
              "literature": "14. Comparison with existing works"}
    out = []
    for key, title in titles.items():
        sec = ex.get(key)
        if not sec:
            continue
        img = f"\n![{key}](figures/{sec['figure']})\n" if sec.get("figure") else ""
        out.append(f"\n## {title}\n{sec['md']}\n{img}")
    return "".join(out)
