"""
Numerical analysis of the ablation study.

Outputs (results/ablation/):
  per_review_metrics.csv     one row per (topic, configuration)
  summary.json               means, 95% CIs, deltas, tests (read by the dashboard)
  ablation_table.md / .tex   main table for the report / paper
  significance.csv           paired Wilcoxon tests (Holm-corrected) + effect sizes
  component_contributions.csv  what each agent adds (drop when removed)
  nli_threshold_sensitivity.csv  robustness of the NLI metric to its threshold
  gap_verification.csv       Skeptic/Verifier precision per topic
  feedback_rounds.csv        critic score per feedback round
  charts/*.png               figures
  REPORT.md                  narrative summary of all of the above
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .benchmark import topic_dir
from .config import CONFIG_LABELS, CONFIG_SHORT, CONFIGS, JUDGE_MODEL, RESULTS_DIR, TOPICS

CONFIG_IDS = [c for c, _ in CONFIGS]
PROPOSED = "M5_full_feedback"

# key, label, higher_is_better, kind ("rate" shown as %, "score", "count")
PRIMARY_METRICS = [
    ("overall_quality_index", "Overall Quality Index", True, "rate"),
    ("reference_validity", "Reference Validity", True, "rate"),
    ("citation_validity", "Citation Validity", True, "rate"),
    ("groundedness", "Citation Groundedness", True, "rate"),
    ("hallucination_rate", "Hallucination Rate", False, "rate"),
    ("nli_entailment_rate", "NLI Entailment", True, "rate"),
    ("source_coverage", "Source Coverage", True, "rate"),
    ("multi_citation_rate", "Cross-Paper Synthesis", True, "rate"),
    ("gap_grounding", "Gap Grounding", True, "rate"),
    ("structure_score", "Structure Score", True, "score"),
]
SECONDARY_METRICS = [
    ("groundedness_lenient", "Groundedness (lenient)", True, "rate"),
    ("contradiction_rate", "Contradiction Rate", False, "rate"),
    ("nli_contradiction_rate", "NLI Contradiction", False, "rate"),
    ("pair_coverage", "Paper-Pair Coverage", True, "rate"),
    ("comparisons_per_1k_words", "Comparisons / 1k words", True, "score"),
    ("gap_specificity", "Gap Specificity", True, "rate"),
    ("gap_novelty", "Gap Novelty", True, "rate"),
    ("gap_citation_rate", "Gap Citation Rate", True, "rate"),
    ("citation_density", "Citation Density", True, "rate"),
    ("overclaim_count", "Over-claims", False, "count"),
    ("word_count", "Words", True, "count"),
    ("llm_tokens", "LLM Tokens", False, "count"),
    ("llm_calls", "LLM Calls", False, "count"),
    ("pipeline_seconds", "Generation Time (s)", False, "count"),
]
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS

# Upstream stages each configuration depends on (for cost accounting).
CONFIG_STAGES = {
    "M0_vanilla": [],
    "M1_retrieval_only": ["taxonomy_abstract"],
    "M2_no_cross_paper": ["analyzed", "taxonomy_full", "gaps_no_cross"],
    "M3_no_skeptic": ["analyzed", "taxonomy_full", "cross", "gaps_full:proposer"],
    "M4_full": ["analyzed", "taxonomy_full", "cross", "gaps_full"],
    "M5_full_feedback": ["analyzed", "taxonomy_full", "cross", "gaps_full", "@M4_full"],
}


# Reads a JSON file; returns the default value if it does not exist or cannot be parsed.
def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

# Works out how much one configuration cost (LLM tokens, LLM calls, seconds).
# It adds up the saved stage costs of every stage that configuration uses; shared stages such as the
# gap proposer are split fairly between configurations.
def _config_cost(folder: Path, config: str) -> Dict[str, float]:
    costs = _read(folder / "stage_costs.json", {}) or {}
    config_costs = _read(folder / "configs" / config / "stage_costs.json", {}) or {}
    total = {"llm_tokens": 0.0, "llm_calls": 0.0, "pipeline_seconds": 0.0}

    # Adds one stage's cost (tokens, calls, seconds), multiplied by a share, to the running totals.
    def add(entry, share: float = 1.0):
        if not entry:
            return
        total["llm_tokens"] += share * (entry.get("input_tokens", 0) + entry.get("output_tokens", 0))
        total["llm_calls"] += share * entry.get("calls", 0)
        total["pipeline_seconds"] += share * entry.get("seconds", 0)

    for stage in CONFIG_STAGES[config]:
        if stage.startswith("@"):
            add((_read(folder / "configs" / stage[1:] / "stage_costs.json", {}) or {}).get("review"))
        elif stage == "gaps_full:proposer":
            gaps = _read(folder / "gaps_full.json", {}) or {}
            full = costs.get("gaps_full") or {}
            usage = (gaps.get("stage_usage") or {}).get("proposer")
            all_tokens = sum(((gaps.get("stage_usage") or {}).get(k) or {}).get("input_tokens", 0)
                             + ((gaps.get("stage_usage") or {}).get(k) or {}).get("output_tokens", 0)
                             for k in ("proposer", "skeptic_verifier"))
            if usage and all_tokens:
                share = (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) / all_tokens
                add(full, share)
            else:
                add(full, 1 / 3)
        else:
            add(costs.get(stage))
    add(config_costs.get("review"))
    return total


# Loads every evaluation result into one flat table: one row per topic and configuration,
# with all metrics and the cost numbers.
def load_rows(topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows = []
    for topic in topics or TOPICS:
        folder = topic_dir(topic)
        for config in CONFIG_IDS:
            evaluation = _read(folder / "configs" / config / "evaluation.json")
            if not evaluation:
                continue
            row = {"topic": topic, "topic_slug": folder.name, "config": config}
            row.update(evaluation["metrics"])
            row.update(_config_cost(folder, config))
            row["_details"] = evaluation.get("details", {})
            rows.append(row)
    return rows


# Returns only the topics that have ALL requested configurations, because paired statistical tests
# need the same topics in every configuration.
def complete_topics(rows: List[Dict[str, Any]], configs: List[str]) -> List[str]:
    """Topics that have every requested configuration (needed for paired tests)."""
    by_topic: Dict[str, set] = {}
    for row in rows:
        by_topic.setdefault(row["topic"], set()).add(row["config"])
    return [t for t, cs in by_topic.items() if set(configs) <= cs]


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

# 95 percent confidence interval of the mean: resample the values many times with replacement,
# compute each mean and take the 2.5th and 97.5th percentiles.
def bootstrap_ci(values: List[float], n_boot: int = 5000, seed: int = 7):
    values = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if len(values) == 0:
        return (None, None)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# Compares two configurations on the same topics with a Wilcoxon signed-rank test.
# Returns the p-value, the mean difference, how many topics each one wins, and an effect size.
def paired_test(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Wilcoxon signed-rank (two-sided) + matched-pairs rank-biserial effect size."""
    from scipy.stats import rankdata, wilcoxon

    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    diffs = np.array([x - y for x, y in pairs], dtype=float)
    out = {
        "n": len(pairs),
        "mean_diff": float(diffs.mean()) if len(diffs) else None,
        "wins": int((diffs > 1e-9).sum()),
        "ties": int((np.abs(diffs) <= 1e-9).sum()),
        "losses": int((diffs < -1e-9).sum()),
        "p_value": None,
        "rank_biserial": None,
    }
    nonzero = diffs[np.abs(diffs) > 1e-9]
    if len(nonzero) == 0:
        out["p_value"] = 1.0
        out["rank_biserial"] = 0.0
        return out
    ranks = rankdata(np.abs(nonzero))
    r_plus, r_minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    out["rank_biserial"] = float((r_plus - r_minus) / (r_plus + r_minus))
    try:
        out["p_value"] = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    except ValueError:
        out["p_value"] = 1.0
    return out


# Holm correction for testing many hypotheses: adjusts the p-values so that finding 'significant'
# results by chance becomes less likely.
def holm(pvalues: List[Optional[float]]) -> List[Optional[float]]:
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    indexed.sort()
    adjusted: List[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    m = len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


# Converts a p-value into stars: * below 0.05, ** below 0.01, *** below 0.001.
def stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------

# Formats a metric for display: rates as percentages, scores with one decimal.
def fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if kind == "rate":
        return f"{100 * value:.1f}%"
    if kind == "score":
        return f"{value:.1f}"
    return f"{value:,.0f}"


# Collects the values of one metric for one configuration over the chosen topics.
def _values(rows, config, key, topics):
    lookup = {(r["topic"], r["config"]): r for r in rows}
    return [lookup.get((t, config), {}).get(key) for t in topics]


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

# Builds the whole result summary of the ablation study:
#  1. loads the rows and keeps the topics that have every configuration,
#  2. writes per_review_metrics.csv,
#  3. computes mean, standard deviation and 95% confidence interval of every metric per configuration,
#  4. runs paired significance tests (M5 against every other configuration, M4 against its ablations) with
#     Holm correction,
#  5. computes the contribution of each component, writes tables, charts, the LaTeX table and REPORT.md.
def run_analysis(topics: Optional[List[str]] = None) -> Dict[str, Any]:
    import csv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    charts = RESULTS_DIR / "charts"
    charts.mkdir(exist_ok=True)

    rows = load_rows(topics)
    if not rows:
        print("No evaluated reviews yet - run 'generate' and 'evaluate' first.")
        return {}

    present_configs = [c for c in CONFIG_IDS if any(r["config"] == c for r in rows)]
    paired_topics = complete_topics(rows, present_configs)
    all_topics = sorted({r["topic"] for r in rows}, key=lambda t: TOPICS.index(t) if t in TOPICS else 99)

    # ---- per-review CSV -------------------------------------------------
    columns = ["topic", "config"] + [k for k, *_ in ALL_METRICS] + ["judge_nli_agreement", "judged_claims", "gap_count"]
    with open(RESULTS_DIR / "per_review_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})

    # ---- descriptive summary (paired topics only, so means are comparable) ----
    basis = paired_topics or all_topics
    summary: Dict[str, Any] = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "topics": basis,
        "n_topics": len(basis),
        "configs": [{"id": c, "label": CONFIG_LABELS[c], "short": CONFIG_SHORT[c]} for c in present_configs],
        "judge_model": JUDGE_MODEL,
        "metrics": [{"key": k, "label": l, "higher_is_better": h, "kind": kind, "primary": (k, l, h, kind) in PRIMARY_METRICS}
                    for k, l, h, kind in ALL_METRICS],
        "table": {},
    }
    for config in present_configs:
        summary["table"][config] = {}
        for key, _label, _hib, _kind in ALL_METRICS:
            values = [v for v in _values(rows, config, key, basis) if v is not None]
            lo, hi = bootstrap_ci(values)
            summary["table"][config][key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "ci_low": lo,
                "ci_high": hi,
                "n": len(values),
            }

    # ---- significance: proposed vs every other config; M4 vs its ablations ----
    comparisons = [(PROPOSED, c) for c in present_configs if c != PROPOSED and PROPOSED in present_configs]
    comparisons += [("M4_full", c) for c in present_configs if c not in {"M4_full", PROPOSED} and "M4_full" in present_configs]
    tests = []
    for key, label, hib, kind in PRIMARY_METRICS + SECONDARY_METRICS[:9]:
        family = []
        for a, b in comparisons:
            result = paired_test(_values(rows, a, key, paired_topics), _values(rows, b, key, paired_topics))
            # Orient so that positive = first config better.
            if not hib and result["mean_diff"] is not None:
                result["mean_diff"] = -result["mean_diff"]
                result["wins"], result["losses"] = result["losses"], result["wins"]
                result["rank_biserial"] = -result["rank_biserial"] if result["rank_biserial"] is not None else None
            family.append({"metric": key, "label": label, "a": a, "b": b, **result})
        # Two pre-specified hypothesis families per metric, each Holm-corrected:
        # (1) proposed M5 vs every other configuration, (2) M4 vs each of its ablations.
        for reference in {entry["a"] for entry in family}:
            group = [entry for entry in family if entry["a"] == reference]
            for entry, p_adj in zip(group, holm([g["p_value"] for g in group])):
                entry["p_holm"] = p_adj
        tests.extend(family)
    summary["tests"] = tests
    with open(RESULTS_DIR / "significance.csv", "w", newline="", encoding="utf-8") as handle:
        fields = ["metric", "a", "b", "n", "mean_diff", "wins", "ties", "losses", "p_value", "p_holm", "rank_biserial"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tests)

    # ---- component contributions ---------------------------------------
    components = [
        ("Evidence pipeline (retrieval→analysis)", "M4_full", "M0_vanilla"),
        ("Paper analysis + gap agents", "M4_full", "M1_retrieval_only"),
        ("Cross-paper synthesis agent", "M4_full", "M2_no_cross_paper"),
        ("Skeptic / Verifier", "M4_full", "M3_no_skeptic"),
        ("Feedback loop", PROPOSED, "M4_full"),
    ]
    contribution_rows = []
    for name, with_c, without_c in components:
        if with_c not in present_configs or without_c not in present_configs:
            continue
        entry = {"component": name, "with": with_c, "without": without_c}
        for key, _label, hib, _kind in PRIMARY_METRICS:
            a = summary["table"][with_c][key]["mean"]
            b = summary["table"][without_c][key]["mean"]
            entry[key] = None if a is None or b is None else (a - b) * (1 if hib else -1)
        contribution_rows.append(entry)
    summary["contributions"] = contribution_rows
    with open(RESULTS_DIR / "component_contributions.csv", "w", newline="", encoding="utf-8") as handle:
        fields = ["component", "with", "without"] + [k for k, *_ in PRIMARY_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(contribution_rows)

    # ---- NLI threshold sensitivity -------------------------------------
    sensitivity = {}
    for config in present_configs:
        sensitivity[config] = {}
        for tau in (0.3, 0.5, 0.7, 0.9):
            per_topic = []
            for row in rows:
                if row["config"] != config or row["topic"] not in basis:
                    continue
                cited = [n for n in row["_details"].get("nli", []) if n.get("cited")]
                if cited:
                    per_topic.append(np.mean([n["entail"] >= tau for n in cited]))
                else:
                    per_topic.append(0.0)
            sensitivity[config][str(tau)] = float(np.mean(per_topic)) if per_topic else None
    summary["nli_sensitivity"] = sensitivity
    with open(RESULTS_DIR / "nli_threshold_sensitivity.csv", "w", newline="", encoding="utf-8") as handle:
        handle.write("config,tau_0.3,tau_0.5,tau_0.7,tau_0.9\n")
        for config, values in sensitivity.items():
            handle.write(config + "," + ",".join(f"{values[t]:.4f}" if values[t] is not None else "" for t in ("0.3", "0.5", "0.7", "0.9")) + "\n")

    # ---- Skeptic / Verifier gap evaluation --------------------------------
    gap_rows = []
    for topic in all_topics:
        gap_eval = _read(topic_dir(topic) / "gap_verification_eval.json", {}) or {}
        full = gap_eval.get("full") or {}
        if full.get("proposed"):
            gap_rows.append({
                "topic": topic,
                "proposed": full.get("proposed"),
                "verified": full.get("verified"),
                "rejection_rate": full.get("rejection_rate"),
                "precision_unfiltered_M3": full.get("precision_unfiltered"),
                "precision_verified_M4": full.get("precision_verified"),
                "skeptic_precision": full.get("skeptic_precision"),
                "precision_no_cross_M2": (gap_eval.get("no_cross") or {}).get("precision_verified"),
            })
    summary["gap_verification"] = {
        "per_topic": gap_rows,
        "mean": {
            k: (float(np.mean([r[k] for r in gap_rows if r.get(k) is not None]))
                if any(r.get(k) is not None for r in gap_rows) else None)
            for k in ("proposed", "verified", "rejection_rate", "precision_unfiltered_M3",
                      "precision_verified_M4", "skeptic_precision", "precision_no_cross_M2")
        },
    }
    if gap_rows:
        with open(RESULTS_DIR / "gap_verification.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(gap_rows)

    # ---- Feedback rounds ---------------------------------------------------
    feedback_rows = []
    for topic in all_topics:
        review = _read(topic_dir(topic) / "configs" / PROPOSED / "review.json", {}) or {}
        for entry in review.get("feedback_log", []) or []:
            if "score" in entry:
                feedback_rows.append({
                    "topic": topic,
                    "round": entry["round"],
                    "critic_score": entry["score"],
                    "accepted": entry.get("accepted"),
                    "issues": sum((entry.get("issue_counts") or {}).values()),
                })
    summary["feedback_rounds"] = feedback_rows
    if feedback_rows:
        with open(RESULTS_DIR / "feedback_rounds.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feedback_rows[0].keys()))
            writer.writeheader()
            writer.writerows(feedback_rows)

    agreement = [r.get("judge_nli_agreement") for r in rows if r.get("judge_nli_agreement") is not None]
    summary["judge_nli_agreement"] = float(np.mean(agreement)) if agreement else None

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_tables(summary)
    try:
        make_charts(summary, rows, charts)
    except Exception as error:  # charts must never block the numbers
        print(f"⚠️ chart rendering failed: {error}")
    write_report(summary)
    print(f"\n✅ Analysis written to {RESULTS_DIR}")
    print((RESULTS_DIR / "ablation_table.md").read_text(encoding="utf-8"))
    return summary


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

# Looks up the corrected p-value for one comparison (metric, configuration A versus B) in the summary.
def _p_for(summary, metric, a, b):
    for t in summary["tests"]:
        if t["metric"] == metric and t["a"] == a and t["b"] == b:
            return t["p_holm"]
    return None


# Writes the ablation tables (Markdown, CSV and the LaTeX table) from the summary:
# means for each configuration, the best value marked, and significance stars.
def write_tables(summary: Dict[str, Any]) -> None:
    configs = [c["id"] for c in summary["configs"]]
    table = summary["table"]
    head = ["Configuration"] + [f"{l} {'↑' if h else '↓'}" for _, l, h, _ in PRIMARY_METRICS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    best = {}
    for key, _l, hib, _k in PRIMARY_METRICS:
        means = [(table[c][key]["mean"], c) for c in configs if table[c][key]["mean"] is not None]
        if means:
            best[key] = (max if hib else min)(means)[1]
    for config in configs:
        cells = [CONFIG_LABELS[config]]
        for key, _l, _h, kind in PRIMARY_METRICS:
            stat = table[config][key]
            cell = fmt(stat["mean"], kind)
            if stat["std"] is not None and kind == "rate":
                cell += f" ± {100 * stat['std']:.1f}"
            elif stat["std"] is not None:
                cell += f" ± {stat['std']:.1f}"
            if config != PROPOSED:
                cell += stars(_p_for(summary, key, PROPOSED, config))
            if best.get(key) == config:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    note = (
        f"\nMean ± SD over {summary['n_topics']} benchmark topics. Stars: paired Wilcoxon signed-rank test of "
        f"M5 vs. that configuration, Holm-corrected (* p<0.05, ** p<0.01, *** p<0.001). "
        f"Bold = best. Judge: {summary['judge_model']}; NLI: DeBERTa-v3-base.\n"
    )
    (RESULTS_DIR / "ablation_table.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")

    # LaTeX
    short = [("overall_quality_index", "OQI"), ("reference_validity", "RefVal"), ("citation_validity", "CitVal"),
             ("groundedness", "Ground."), ("hallucination_rate", "Halluc."), ("nli_entailment_rate", "NLI"),
             ("source_coverage", "Cover."), ("multi_citation_rate", "Synth."), ("gap_grounding", "GapG."),
             ("structure_score", "Struct.")]
    kinds = {k: kind for k, _l, _h, kind in PRIMARY_METRICS}
    arrows = {k: ("$\\uparrow$" if h else "$\\downarrow$") for k, _l, h, _k in PRIMARY_METRICS}
    tex = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        f"\\caption{{Ablation study on {summary['n_topics']} benchmark topics (mean over topics). "
        "Stars: paired Wilcoxon vs. the proposed system, Holm-corrected.}",
        "\\begin{tabular}{l" + "c" * len(short) + "}", "\\toprule",
        "\\textbf{Configuration} & " + " & ".join(f"\\textbf{{{s}}} {arrows[k]}" for k, s in short) + " \\\\",
        "\\midrule",
    ]
    for config in configs:
        cells = []
        for key, _s in short:
            value = table[config][key]["mean"]
            text = "--" if value is None else (f"{100 * value:.1f}" if kinds[key] == "rate" else f"{value:.1f}")
            if config != PROPOSED:
                text += stars(_p_for(summary, key, PROPOSED, config)).replace("*", "$^*$")
            if best.get(key) == config:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        name = CONFIG_LABELS[config].replace("&", "\\&")
        tex.append(f"{name} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\label{tab:ablation}", "\\end{table*}"]
    (RESULTS_DIR / "ablation_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    summary["latex_table"] = "\n".join(tex)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


# ----------------------------------------------------------------------
# Charts (matplotlib, static PNG for slides / report)
# ----------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b2"
HIGHLIGHT = "#2a78d6"   # proposed system
SECOND = "#eb6834"      # M4 in two-series line charts
DIVERGING_NEG = "#e34948"
DIVERGING_POS = "#2a78d6"
DIVERGING_MID = "#f0efec"


# Applies a common clean look to a chart (no top/right/left lines, light grid, small ticks).
def _style(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Draws all result figures with matplotlib and saves them as PNG files
# (groundedness, synthesis, judge scores, radar, contributions, feedback loop, topic heatmap, boxplots ...).
# These images appear in the Evaluation tab.
def make_charts(summary: Dict[str, Any], rows: List[Dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150, "savefig.bbox": "tight"})
    configs = [c["id"] for c in summary["configs"]]
    table = summary["table"]

    # Draws one bar chart panel for a single metric: one bar per configuration, best one highlighted.
    def bar_panel(ax, key, label, hib, kind):
        means, lows, highs = [], [], []
        for c in configs:
            s = table[c][key]
            m = s["mean"] if s["mean"] is not None else 0.0
            scale = 100 if kind == "rate" else 1
            means.append(m * scale)
            lows.append((m - (s["ci_low"] if s["ci_low"] is not None else m)) * scale)
            highs.append(((s["ci_high"] if s["ci_high"] is not None else m) - m) * scale)
        y = np.arange(len(configs))[::-1]
        colors = [HIGHLIGHT if c == PROPOSED else NEUTRAL for c in configs]
        ax.barh(y, means, color=colors, height=0.62, edgecolor="white", linewidth=2)
        ax.errorbar(means, y, xerr=[lows, highs], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5)
        ax.set_yticks(y, [CONFIG_SHORT[c] for c in configs])
        unit = "%" if kind == "rate" else ""
        for yi, m, h in zip(y, means, highs):
            ax.text(m + h, yi, f"  {m:.1f}{unit}", va="center", ha="left", fontsize=8, color=INK)
        ax.set_title(f"{label} {'↑' if hib else '↓'}", fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_xlim(0, max(100, max(m + h for m, h in zip(means, highs)) * 1.18) if kind in {"rate", "score"} else None)
        _style(ax)

    # 1. Overall quality index
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bar_panel(ax, "overall_quality_index", "Overall Quality Index (mean, 95% CI)", True, "rate")
    fig.savefig(out / "overall_quality_index.png")
    plt.close(fig)

    # 2. Small multiples of primary metrics
    panels = [m for m in PRIMARY_METRICS if m[0] != "overall_quality_index"]
    cols = 3
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(13, 3.2 * rows_n))
    for ax, (key, label, hib, kind) in zip(axes.ravel(), panels):
        bar_panel(ax, key, label, hib, kind)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Ablation study — primary metrics (mean ± 95% bootstrap CI)", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "ablation_primary_metrics.png")
    plt.close(fig)

    # 3. Delta heatmap: proposed minus each config (oriented: positive = proposed better)
    others = [c for c in configs if c != PROPOSED]
    if PROPOSED in configs and others:
        keys = [m for m in PRIMARY_METRICS]
        matrix = np.zeros((len(others), len(keys)))
        annotations = []
        for i, c in enumerate(others):
            row_notes = []
            for j, (key, _l, hib, kind) in enumerate(keys):
                a, b = table[PROPOSED][key]["mean"], table[c][key]["mean"]
                d = 0.0 if a is None or b is None else (a - b) * (1 if hib else -1)
                if kind == "score":
                    d = d / 100.0
                matrix[i, j] = d * 100
                row_notes.append(f"{d * 100:+.1f}{stars(_p_for(summary, key, PROPOSED, c))}")
            annotations.append(row_notes)
        limit = max(5.0, float(np.abs(matrix).max()))
        cmap = LinearSegmentedColormap.from_list("div", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])
        fig, ax = plt.subplots(figsize=(1.15 * len(keys) + 2.5, 0.62 * len(others) + 1.8))
        ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(keys)), [l for _, l, _, _ in keys], rotation=35, ha="right", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(others)), [f"vs {CONFIG_SHORT[c]}" for c in others], fontsize=9, color=INK_2)
        for i in range(len(others)):
            for j in range(len(keys)):
                ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=8, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title("Gain of the proposed system (M5) over each configuration, in points\n"
                     "blue = M5 better, red = M5 worse; stars = Holm-corrected Wilcoxon",
                     fontsize=10, color=INK, loc="left")
        fig.savefig(out / "delta_heatmap.png")
        plt.close(fig)

    # 4. Feedback rounds
    feedback = summary.get("feedback_rounds") or []
    if feedback:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        by_topic: Dict[str, list] = {}
        for r in feedback:
            by_topic.setdefault(r["topic"], []).append(r)
        max_round = max(r["round"] for r in feedback)
        kept_per_round = {k: [] for k in range(max_round + 1)}
        for topic, entries in by_topic.items():
            entries.sort(key=lambda r: r["round"])
            score, xs, ys = None, [], []
            for e in entries:
                if e["round"] == 0 or e["accepted"]:
                    score = e["critic_score"]
                xs.append(e["round"])
                ys.append(score)
            for k in range(max_round + 1):
                kept_per_round[k].append(ys[min(k, len(ys) - 1)])
            ax.plot(xs, ys, color=NEUTRAL, linewidth=1, marker="o", markersize=3)
        mean_line = [np.mean(kept_per_round[k]) for k in range(max_round + 1)]
        ax.plot(range(max_round + 1), mean_line, color=HIGHLIGHT, linewidth=2.5, marker="o", markersize=7, label="mean")
        for k, m in enumerate(mean_line):
            ax.text(k, m, f"  {m:.1f}", color=INK, fontsize=9, va="bottom")
        ax.set_xticks(range(max_round + 1), ["draft"] + [f"round {k}" for k in range(1, max_round + 1)])
        ax.set_ylabel("critic score (kept draft)", color=INK_2, fontsize=9)
        ax.set_title("Feedback loop: critic score per round (gray = topics, blue = mean)", fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.grid(axis="y", color=GRID)
        ax.grid(axis="x", visible=False)
        fig.savefig(out / "feedback_rounds.png")
        plt.close(fig)

    # 5. Gap verification: unfiltered (M3) vs verified (M4)
    gap = (summary.get("gap_verification") or {}).get("per_topic") or []
    gap = [g for g in gap if g.get("precision_unfiltered_M3") is not None and g.get("precision_verified_M4") is not None]
    if gap:
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(gap) + 1.6))
        y = np.arange(len(gap))[::-1]
        for yi, g in zip(y, gap):
            a, b = 100 * g["precision_unfiltered_M3"], 100 * g["precision_verified_M4"]
            ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.scatter([a], [yi], color=NEUTRAL, s=60, zorder=2, edgecolor="white", linewidth=2)
            ax.scatter([b], [yi], color=HIGHLIGHT, s=60, zorder=3, edgecolor="white", linewidth=2)
        ax.set_yticks(y, [g["topic"][:42] for g in gap], fontsize=8)
        ax.set_xlim(-2, 105)
        ax.set_xlabel("gap precision (% judged supported by source text)", fontsize=9, color=INK_2)
        ax.scatter([], [], color=NEUTRAL, s=40, label="proposer only (M3)")
        ax.scatter([], [], color=HIGHLIGHT, s=40, label="after Skeptic + Verifier (M4)")
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        _style(ax)
        fig.savefig(out / "gap_verification.png")
        plt.close(fig)

    # 6. NLI threshold sensitivity (M5 and M4 highlighted, others gray)
    sens = summary.get("nli_sensitivity") or {}
    if sens:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        taus = ["0.3", "0.5", "0.7", "0.9"]
        for c in configs:
            values = [100 * (sens[c][t] or 0) for t in taus]
            color = HIGHLIGHT if c == PROPOSED else SECOND if c == "M4_full" else NEUTRAL
            width = 2.5 if c in {PROPOSED, "M4_full"} else 1.2
            ax.plot([float(t) for t in taus], values, color=color, linewidth=width, marker="o", markersize=5 if width > 2 else 3)
            ax.text(0.92, values[-1], CONFIG_SHORT[c], fontsize=8, color=INK_2, va="center")
        ax.set_xlabel("NLI entailment threshold τ", fontsize=9, color=INK_2)
        ax.set_ylabel("% cited sentences entailed", fontsize=9, color=INK_2)
        ax.set_xlim(0.25, 1.05)
        ax.set_title("Robustness: ranking is stable across NLI thresholds", fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.grid(axis="y", color=GRID)
        fig.savefig(out / "nli_threshold_sensitivity.png")
        plt.close(fig)


# ----------------------------------------------------------------------
# Narrative report
# ----------------------------------------------------------------------

# Writes REPORT.md: the main table, the statistics and the key takeaways in one readable text file.
def write_report(summary: Dict[str, Any]) -> None:
    table_md = (RESULTS_DIR / "ablation_table.md").read_text(encoding="utf-8")
    lines = [
        "# Ablation Study — Results",
        "",
        f"*Generated {summary['generated_at']} · {summary['n_topics']} topics · judge `{summary['judge_model']}` · NLI `cross-encoder/nli-deberta-v3-base`*",
        "",
        "All numbers below are computed automatically from the generated reviews (no manual labels). "
        "See `EVALUATION.md` for the definition of every metric.",
        "",
        "## Main table",
        "",
        table_md,
        "",
        "## What each component contributes",
        "",
        "Difference in points (oriented so positive = the component helps).",
        "",
        "| Component | " + " | ".join(l for _, l, _, _ in PRIMARY_METRICS) + " |",
        "|---|" + "---|" * len(PRIMARY_METRICS),
    ]
    for c in summary.get("contributions", []):
        cells = []
        for key, _l, _h, kind in PRIMARY_METRICS:
            v = c.get(key)
            cells.append("–" if v is None else (f"{100 * v:+.1f}" if kind == "rate" else f"{v:+.1f}"))
        lines.append(f"| {c['component']} | " + " | ".join(cells) + " |")

    gv = (summary.get("gap_verification") or {}).get("mean") or {}
    if gv.get("precision_unfiltered_M3") is not None:
        lines += [
            "",
            "## Skeptic / Verifier (research-gap reliability)",
            "",
            f"- Mean candidate gaps proposed per topic: {gv['proposed']:.1f}; kept after verification: {gv['verified']:.1f} "
            f"(rejection rate {100 * (gv['rejection_rate'] or 0):.1f}%).",
            f"- Gap precision without Skeptic (M3): **{100 * gv['precision_unfiltered_M3']:.1f}%** → "
            f"with Skeptic + Verifier (M4): **{100 * (gv['precision_verified_M4'] or 0):.1f}%**.",
        ]
        if gv.get("skeptic_precision") is not None:
            lines.append(f"- Skeptic precision (rejected gaps that were truly unsupported): **{100 * gv['skeptic_precision']:.1f}%**.")

    fb = summary.get("feedback_rounds") or []
    if fb:
        draft = [r["critic_score"] for r in fb if r["round"] == 0]
        accepted = [r for r in fb if r["round"] > 0 and r["accepted"]]
        lines += [
            "",
            "## Feedback loop",
            "",
            f"- Mean critic score of the M4 draft: {np.mean(draft):.1f}.",
            f"- Revisions accepted: {len(accepted)} of {sum(1 for r in fb if r['round'] > 0)} attempted "
            "(a revision is kept only if the critic score improves).",
        ]

    if summary.get("judge_nli_agreement") is not None:
        lines += [
            "",
            "## Evaluator reliability",
            "",
            f"- Agreement between the LLM judge and the independent NLI model on sampled claims: "
            f"{100 * summary['judge_nli_agreement']:.1f}%.",
            "- NLI entailment rates at several thresholds are in `nli_threshold_sensitivity.csv`; "
            "the configuration ranking should be read as robust only if it holds across thresholds.",
        ]

    lines += [
        "",
        "## Figures",
        "",
        "![Overall quality](charts/overall_quality_index.png)",
        "",
        "![Primary metrics](charts/ablation_primary_metrics.png)",
        "",
        "![Delta heatmap](charts/delta_heatmap.png)",
        "",
        "![Gap verification](charts/gap_verification.png)",
        "",
        "![Feedback rounds](charts/feedback_rounds.png)",
        "",
        "![NLI sensitivity](charts/nli_threshold_sensitivity.png)",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# Prints a progress table: for each topic which stages exist (papers, extracted, analyzed, taxonomy, cross, gaps)
# and, for each configuration, whether it is generated (G) or evaluated (E).
def print_status(topics: Optional[List[str]] = None) -> None:
    stages = ["papers", "extracted", "analyzed", "taxonomy_full", "cross", "gaps_full", "gaps_no_cross", "taxonomy_abstract"]
    print(f"{'topic':44} " + " ".join(s[:6] for s in stages) + " | " + " ".join(c[:2] for c in CONFIG_IDS))
    for topic in topics or TOPICS:
        folder = topic_dir(topic)
        marks = ["✓" if (folder / f"{s}.json").exists() else "·" for s in stages]
        cfg = []
        for c in CONFIG_IDS:
            if (folder / "configs" / c / "evaluation.json").exists():
                cfg.append("E")
            elif (folder / "configs" / c / "review.json").exists():
                cfg.append("G")
            else:
                cfg.append("·")
        print(f"{topic[:44]:44} " + "      ".join(marks) + " | " + "  ".join(cfg))
    print("\nG = generated, E = evaluated")
