"""Publication figures for the ablation study (matplotlib, PNG @ 200 dpi)."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from evaluation.stats import METRICS, bootstrap_ci

# validated categorical order (dataviz reference palette, light mode), fixed per config
CONFIG_COLORS = {"M0": "#2a78d6", "M1": "#eb6834", "M2": "#1baf7a", "M3": "#eda100",
                 "M4": "#e87ba4", "M5": "#4a3aa7",
                 # baseline group shows M0, B1, B2, M1, M5 together: B1/B2 take the free slots
                 "B1": "#1baf7a", "B2": "#eda100",
                 # prompt group (P1-P5) and preference group (M4 vs U1) are plotted separately
                 "P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a", "P4": "#eda100", "P5": "#e87ba4",
                 "U1": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUES = ["#eaf2fc", "#c6dbf6", "#94bbee", "#5e97e3", "#2a78d6", "#1d5aa6", "#123d73"]


def _style(ax, title=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def ablation_bars(df, metrics, name, title):
    configs = sorted(df["config_id"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        label, higher, pct = METRICS[m]
        means, los, his = [], [], []
        for c in configs:
            x = df[df.config_id == c][m].astype(float).values
            mu = np.nanmean(x) if len(x) else np.nan
            lo, hi = bootstrap_ci(x)
            k = 100 if pct else 1
            means.append(mu * k)
            los.append((mu - lo) * k if not np.isnan(lo) else 0)
            his.append((hi - mu) * k if not np.isnan(hi) else 0)
        xs = np.arange(len(configs))
        ax.bar(xs, means, width=0.7, color=[CONFIG_COLORS[c] for c in configs],
               edgecolor="white", linewidth=2, yerr=[los, his], error_kw={"ecolor": INK2, "elinewidth": 1,
                                                                           "capsize": 2})
        ax.set_xticks(xs, configs)
        arrow = "↑" if higher else ("↓" if higher is False else "")
        _style(ax, f"{label} {arrow}", "%" if pct else None)
        best = int(np.nanargmax(means) if higher else np.nanargmin(means)) if higher is not None else None
        if best is not None:
            ax.annotate(f"{means[best]:.1f}", (best, means[best] + his[best]), ha="center",
                        va="bottom", fontsize=8, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONFIG_COLORS[c]) for c in configs]
    fig.legend(handles, [config.CONFIGS[c]["name"] for c in configs], loc="lower center",
               ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return _save(fig, name)


def radar(df, metrics, name):
    configs = sorted(df["config_id"].unique())
    vals = {}
    for c in configs:
        row = []
        for m in metrics:
            x = df[df.config_id == c][m].astype(float).mean()
            if m.startswith("judge_") and not METRICS[m][2]:
                x = x / 10
            if METRICS[m][1] is False:
                x = 1 - x
            row.append(np.clip(x, 0, 1))
        vals[c] = row
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, polar=True)
    for c in configs:
        v = vals[c] + vals[c][:1]
        lw = 2.6 if c == config.PROPOSED else 1.4
        ax.plot(ang, v, color=CONFIG_COLORS[c], linewidth=lw, label=config.CONFIGS[c]["name"])
        if c == config.PROPOSED:
            ax.fill(ang, v, color=CONFIG_COLORS[c], alpha=0.12)
    labels = []
    for m in metrics:
        lab = METRICS[m][0].replace("Judge: ", "")
        labels.append(("1 − " + lab) if METRICS[m][1] is False else lab)
    ax.set_xticks(ang[:-1], labels, fontsize=8, color=INK2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1"], fontsize=7, color=INK2)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=8)
    ax.set_title("Multi-metric profile (normalized, higher = better)", fontsize=11, color=INK,
                 fontweight="bold", pad=18)
    return _save(fig, name)


def contributions(contrib, name, metric="overall_quality_index"):
    if contrib is None or contrib.empty or "metric" not in contrib:
        return None   # needs >= 2 complete topics
    sub = contrib[contrib.metric == metric].copy()
    if sub.empty:
        return None
    pct = METRICS[metric][2]
    k = 100 if pct else 1
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ys = np.arange(len(sub))[::-1]
    deltas = sub["delta"].values * k
    lo = np.array([d - c[0] * k if not np.isnan(c[0]) else 0 for d, c in zip(deltas, sub["ci"])])
    hi = np.array([c[1] * k - d if not np.isnan(c[1]) else 0 for d, c in zip(deltas, sub["ci"])])
    ax.barh(ys, deltas, color="#2a78d6", height=0.6, xerr=[lo, hi],
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    span = max(1e-6, float(np.max(np.abs(deltas) + np.maximum(lo, hi))))
    for y, d, p, l, h in zip(ys, deltas, sub["p"].values, lo, hi):
        star = "*" if p < 0.05 else ""
        x = d + h + 0.03 * span if d >= 0 else d - l - 0.03 * span   # past the CI whisker
        ax.text(x, y, f"{d:+.1f}{star}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8, color=INK)
    ax.set_xlim(min(0, float(np.min(deltas - lo))) - 0.18 * span,
                max(0, float(np.max(deltas + hi))) + 0.18 * span)
    ax.set_yticks(ys, sub["component"].values, fontsize=8.5)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Δ {METRICS[metric][0]} ({'pp' if pct else 'points'}) — * p<0.05 (Wilcoxon)",
                  color=INK2, fontsize=9)
    ax.set_title("Contribution of each component", loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def feedback_curve(runs, name):
    series = []
    for r in runs:
        h = r.get("feedback_history") or []
        if not h:
            continue
        series.append([x.get("unsupported_or_uncited", np.nan) for x in h])
    if not series:
        return None
    L = max(len(s) for s in series)
    arr = np.full((len(series), L), np.nan)
    for i, s in enumerate(series):
        arr[i, :len(s)] = s
        arr[i, len(s):] = s[-1]
    mu = np.nanmean(arr, axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xs = np.arange(L)
    labels = ["draft"] + [f"after round {i}" for i in range(1, L)]
    for row in arr:
        ax.plot(xs, row, color=GRID, linewidth=1)
    ax.plot(xs, mu, color="#4a3aa7", linewidth=2.4, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2)
    for x, y in zip(xs, mu):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(xs, labels[:L])
    _style(ax, "Feedback loop: unsupported/uncited sentences", "sentences per review")
    fig.tight_layout()
    return _save(fig, name)


def topic_heatmap(df, metric, name):
    piv = df.pivot_table(index="topic", columns="config_id", values=metric)
    pct = METRICS[metric][2]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUES)
    if METRICS[metric][1] is False:
        cmap = cmap.reversed()
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(piv) + 1.4))
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto", vmin=0, vmax=1 if pct else None)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [t[:48] for t in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            txt = f"{100 * v:.0f}" if pct else f"{v:.1f}"
            norm = im.norm(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if norm > 0.55 else INK)
    ax.set_title(f"{METRICS[metric][0]} per topic{' (%)' if pct else ''}", loc="left", fontsize=11,
                 color=INK, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return _save(fig, name)


def retrieval_bars(ret_df, name):
    if ret_df is None or ret_df.empty:
        return None
    metrics = ["P@5", "nDCG@10", "MRR"]
    agg = ret_df.groupby("ranker")[metrics].mean().reindex(["hybrid", "semantic", "tfidf", "title"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(metrics))
    w = 0.2
    colors = ["#4a3aa7", "#2a78d6", "#eb6834", "#1baf7a"]
    for i, (rk, row) in enumerate(agg.iterrows()):
        ax.bar(xs + (i - 1.5) * w, row.values, width=w, color=colors[i], edgecolor="white",
               linewidth=1.5, label={"hybrid": "Hybrid (60/20/15/5)", "semantic": "Semantic only",
                                     "tfidf": "TF-IDF only", "title": "Title match only"}[rk])
    ax.set_xticks(xs, metrics)
    ax.set_ylim(0, 1.05)
    _style(ax, "Retrieval ablation (silver relevance labels)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    return _save(fig, name)


def cost_quality(df, name):
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for c in sorted(df.config_id.unique()):
        sub = df[df.config_id == c]
        ax.scatter(sub["total_tokens"].mean() / 1000, 100 * sub["overall_quality_index"].mean(),
                   s=90, color=CONFIG_COLORS[c], edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(c, (sub["total_tokens"].mean() / 1000, 100 * sub["overall_quality_index"].mean()),
                    textcoords="offset points", xytext=(7, 4), fontsize=9, color=INK)
    _style(ax, "Quality vs. cost", "Grounded Quality Index (%)")
    ax.set_xlabel("tokens per review (thousands)", color=INK2, fontsize=9)
    ax.xaxis.grid(True, color=GRID)
    fig.tight_layout()
    return _save(fig, name)


def distribution(df, metric, name):
    configs = sorted(df.config_id.unique())
    data = [df[df.config_id == c][metric].astype(float).dropna().values * (100 if METRICS[metric][2] else 1)
            for c in configs]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], configs):
        patch.set_facecolor(CONFIG_COLORS[c])
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK2)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d, s=12, color=INK, zorder=3)
    ax.set_xticks(range(1, len(configs) + 1), configs)
    _style(ax, f"{METRICS[metric][0]} across topics", "%" if METRICS[metric][2] else None)
    fig.tight_layout()
    return _save(fig, name)
