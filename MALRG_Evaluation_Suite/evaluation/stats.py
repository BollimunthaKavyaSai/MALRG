"""
Numerical analysis of the ablation study.

For every metric and configuration:
  * mean, standard deviation, 95% bootstrap confidence interval (over topics)
  * absolute and relative difference to the proposed system
  * paired Wilcoxon signed-rank test (proposed vs. each ablation), Holm-Bonferroni corrected
  * paired effect sizes: Cohen's d_z and matched-pairs rank-biserial correlation
  * win rate of the proposed system across topics
  * Friedman omnibus test across all configurations + Kendall's W
  * component contribution (drop in performance when a component is removed)
"""

import numpy as np
import pandas as pd
from scipy import stats

import config

# metric -> (display name, higher_is_better, is_percentage)
METRICS = {
    "citation_groundedness": ("Citation Groundedness", True, True),
    "citation_precision": ("Citation Precision", True, True),
    "hallucination_rate": ("Hallucination Rate", False, True),
    "contradiction_rate": ("Contradiction Rate", False, True),
    "uncited_claim_rate": ("Uncited Claim Rate", False, True),
    "fabricated_reference_rate": ("Fabricated Reference Rate", False, True),
    "source_coverage": ("Source Coverage", True, True),
    "multi_source_synthesis": ("Multi-Source Synthesis", True, True),
    "cross_paper_comparison": ("Cross-Paper Comparison", True, True),
    "citation_balance": ("Citation Balance", True, False),
    "judge_claim_support": ("LLM-Judge Claim Support", True, True),
    "judge_synthesis": ("Judge: Synthesis", True, False),
    "judge_critical_analysis": ("Judge: Critical Analysis", True, False),
    "judge_gap_quality": ("Judge: Gap Quality", True, False),
    "judge_coverage": ("Judge: Coverage", True, False),
    "judge_coherence": ("Judge: Coherence", True, False),
    "judge_faithfulness": ("Judge: Faithfulness", True, False),
    "judge_overall": ("Judge: Overall (1-10)", True, False),
    "judge_gap_precision": ("Verified Gap Precision", True, True),
    "structure_score": ("Structural Integrity", True, True),
    "synthesis_indicators_per_1k": ("Synthesis Indicators /1k words", True, False),
    "gap_indicators": ("Research-Gap Indicators", True, False),
    "certainty_flags": ("Certainty / Overclaim Flags", False, False),
    "redundancy": ("Redundancy", False, True),
    "word_count": ("Word Count", None, False),
    "overall_quality_index": ("Grounded Quality Index (GQI)", True, True),
    "soft_groundedness": ("Soft Groundedness (mean entailment)", True, True),
    "numeric_fidelity": ("Numeric Fidelity", True, True),
    "semantic_alignment": ("Semantic Source Alignment", True, False),
    "source_content_recall": ("Source Content Recall", True, True),
    "topic_relevance": ("Topic Relevance", True, False),
    "citation_density": ("Citations per 100 words", True, False),
    "lexical_diversity": ("Lexical Diversity (MTLD)", True, False),
    "readability_fk_grade": ("Flesch-Kincaid Grade", False, False),
    "flesch_reading_ease": ("Flesch Reading Ease", True, False),
    "pref_compliance_rate": ("Preference Compliance", True, True),
    "preference_compliance": ("Preference Compliance", True, True),
    "total_tokens": ("Tokens per Review", False, False),
    "llm_calls": ("LLM Calls per Review", False, False),
}

HEADLINE = ["citation_groundedness", "citation_precision", "hallucination_rate",
            "fabricated_reference_rate", "source_coverage", "multi_source_synthesis",
            "cross_paper_comparison", "judge_gap_precision", "judge_overall", "overall_quality_index"]

GQI_COMPONENTS = ["citation_groundedness", "citation_precision", "1-hallucination_rate",
                  "source_coverage", "multi_source_synthesis", "judge_gap_precision",
                  "judge_overall/10", "structure_score"]


def add_quality_index(df):
    """Grounded Quality Index: unweighted mean of 8 normalized [0,1] quality signals."""
    parts = [
        df["citation_groundedness"], df["citation_precision"], 1 - df["hallucination_rate"],
        df["source_coverage"], df["multi_source_synthesis"], df["judge_gap_precision"],
        df["judge_overall"] / 10.0, df["structure_score"],
    ]
    df["overall_quality_index"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return df


def bootstrap_ci(x, n=config.BOOTSTRAP_SAMPLES, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def paired_tests(df, metric, proposed=config.PROPOSED):
    higher = METRICS[metric][1]
    piv = df.pivot_table(index="topic", columns="config_id", values=metric)
    rows = []
    if proposed not in piv:
        return pd.DataFrame()
    for c in piv.columns:
        if c == proposed:
            continue
        pair = piv[[proposed, c]].dropna()
        d = pair[proposed] - pair[c]
        n = len(d)
        if n < 2:
            continue
        if np.allclose(d, 0):
            p = 1.0
        else:
            try:
                p = stats.wilcoxon(pair[proposed], pair[c], zero_method="wilcox",
                                   alternative="two-sided").pvalue
            except ValueError:
                p = 1.0
        sd = d.std(ddof=1)
        dz = float(d.mean() / sd) if sd > 0 else (np.inf if d.mean() != 0 else 0.0)
        nz = d[d != 0]
        ranks = stats.rankdata(np.abs(nz))
        rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
        if higher is None:
            wins = np.nan
        else:
            better = d > 0 if higher else d < 0
            wins = float(better.mean())
        rows.append({"metric": metric, "baseline": c, "n_topics": n,
                     "proposed_mean": float(pair[proposed].mean()), "baseline_mean": float(pair[c].mean()),
                     "delta": float(d.mean()),
                     "rel_delta_pct": float(100 * d.mean() / abs(pair[c].mean())) if pair[c].mean() else np.nan,
                     "wilcoxon_p": float(p), "cohens_dz": dz, "rank_biserial": rbc, "win_rate": wins})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["wilcoxon_p"].values)
        out["significant"] = out["p_holm"] < config.STAT_ALPHA
    return out


def friedman(df, metric):
    piv = df.pivot_table(index="topic", columns="config_id", values=metric).dropna()
    if piv.shape[0] < 3 or piv.shape[1] < 3:
        return {}
    try:
        stat, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    except ValueError:
        return {}
    k, n = piv.shape[1], piv.shape[0]
    w = stat / (n * (k - 1))
    higher = METRICS[metric][1]
    ranks = piv.rank(axis=1, ascending=not higher if higher is not None else True).mean()
    return {"metric": metric, "chi2": float(stat), "p": float(p), "kendall_w": float(w),
            **{f"mean_rank_{c}": float(ranks[c]) for c in piv.columns}}


def summary_table(df):
    rows = []
    for c in sorted(df["config_id"].unique()):
        sub = df[df["config_id"] == c]
        row = {"config_id": c, "config": config.CONFIGS[c]["name"], "n_topics": len(sub)}
        for m in METRICS:
            if m not in sub:
                continue
            x = sub[m].astype(float)
            lo, hi = bootstrap_ci(x)
            row[f"{m}_mean"] = float(x.mean())
            row[f"{m}_std"] = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def component_contributions(df):
    """Drop in each headline metric when a component is removed (paired over topics)."""
    pairs = {
        "Retrieval grounding (M1 vs M0)": ("M1", "M0"),
        "Evidence analysis + gap triad (M2 vs M1)": ("M2", "M1"),
        "Cross-paper synthesis (M4 vs M2)": ("M4", "M2"),
        "Adversarial gap verification (M4 vs M3)": ("M4", "M3"),
        "Verification-guided feedback (M5 vs M4)": ("M5", "M4"),
    }
    rows = []
    for label, (a, b) in pairs.items():
        for m in HEADLINE + ["judge_synthesis", "judge_gap_quality", "judge_critical_analysis"]:
            if m not in df:
                continue
            piv = df.pivot_table(index="topic", columns="config_id", values=m)
            if a not in piv or b not in piv:
                continue
            pair = piv[[a, b]].dropna()
            if len(pair) < 2:
                continue
            d = pair[a] - pair[b]
            try:
                p = stats.wilcoxon(pair[a], pair[b]).pvalue if not np.allclose(d, 0) else 1.0
            except ValueError:
                p = 1.0
            rows.append({"component": label, "metric": m, "with": a, "without": b,
                         "delta": float(d.mean()), "ci": bootstrap_ci(d.values), "p": float(p)})
    return pd.DataFrame(rows)


def fmt(metric, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if METRICS.get(metric, ("", True, False))[2]:
        return f"{100 * value:.1f}%"
    if metric in ("word_count", "total_tokens", "llm_calls", "gap_indicators"):
        return f"{value:.0f}"
    return f"{value:.2f}"
