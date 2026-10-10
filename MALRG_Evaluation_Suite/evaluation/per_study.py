"""
Per-study analysis of the 50 source papers (10 topics x 5 frozen full-text studies).

For every study: bibliographic data, full-text extraction, what the Paper Analysis Agent extracted
(evidence fidelity, field completeness), the verified research gaps it supports, and how accurately
each configuration cited it (sentences citing it and the share entailed by that study's own text).

    python -m evaluation.per_study      ->  results/tables/per_study_analysis.csv
                                            results/STUDIES.md
"""

import json

import numpy as np
import pandas as pd

import config
from evaluation.metrics import FIELDS
from malrg.retrieval import benchmark_path, slugify

CONFIGS = ["M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5", "U1"]
ABLATION = ["M1", "M2", "M3", "M4", "M5"]


def _load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _filled(a):
    n = 0
    for f in FIELDS:
        v = a.get(f)
        txt = " ".join(map(str, v)) if isinstance(v, list) else str(v or "")
        if txt.strip() and "not reported" not in txt.lower():
            n += 1
    return n / len(FIELDS)


def build():
    rows = []
    for t_idx, topic in enumerate(config.TOPICS, 1):
        bench = _load(benchmark_path(topic))
        if not bench:
            continue
        d = config.RUNS_DIR / slugify(topic)
        m4 = _load(d / "M4.json") or {}
        analyses = {a["ref_id"]: a for a in (m4.get("analyses") or [])}
        gaps = (m4.get("gap_result") or {}).get("final") or []
        evals = {c: _load(d / f"{c}_eval.json") or {} for c in CONFIGS}
        for p in bench["papers"]:
            r = p["ref_id"]
            a = analyses.get(r, {})
            row = {
                "study_id": f"T{t_idx:02d}-S{r}",
                "topic": topic,
                "title": p["title"],
                "year": p.get("year"),
                "venue": p.get("venue"),
                "source": p.get("source"),
                "doi_or_url": p.get("doi") or p.get("url"),
                "fulltext_words": p.get("fulltext_words"),
                "sections_detected": len(p.get("sections") or {}),
                "method": a.get("method"),
                "dataset": a.get("dataset"),
                "key_results": "; ".join(map(str, (a.get("key_results") or [])[:3])),
                "metrics_reported": "; ".join(map(str, (a.get("metrics") or [])[:4])),
                "limitations": "; ".join(map(str, (a.get("limitations") or [])[:3])),
                "evidence_quotes_verified": len(a.get("evidence") or []),
                "evidence_fidelity": a.get("evidence_fidelity"),
                "field_completeness": _filled(a) if a else np.nan,
                "verified_gaps_supported": sum(1 for g in gaps if r in (g.get("supporting_papers") or [])),
            }
            for c in CONFIGS:
                sents = [s for s in (evals[c].get("sentences") or []) if r in (s.get("citations") or [])]
                sup = [s for s in sents if (s.get("per_ref_entail") or {}).get(str(r), 0.0) >= config.NLI_ENTAIL_THRESHOLD
                       or (s.get("per_ref_entail") or {}).get(r, 0.0) >= config.NLI_ENTAIL_THRESHOLD]
                row[f"{c}_sentences_citing"] = len(sents)
                row[f"{c}_citation_accuracy"] = len(sup) / len(sents) if sents else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / "per_study_analysis.csv", index=False)
    tests = study_tests(df)
    tests.to_csv(config.TABLES_DIR / "per_study_tests.csv", index=False)
    figs = study_figures(df)
    write_markdown(df, tests, figs)
    return df


def study_tests(df):
    from scipy import stats as sps
    from evaluation.stats import holm
    rows = []
    for c in [x for x in CONFIGS if x != "M5"]:
        pair = df[["M5_citation_accuracy", f"{c}_citation_accuracy"]].dropna()
        if len(pair) < 5:
            continue
        a, b = pair.iloc[:, 0], pair.iloc[:, 1]
        d = a - b
        try:
            p = sps.wilcoxon(a, b).pvalue if not np.allclose(d, 0) else 1.0
        except ValueError:
            p = 1.0
        rows.append({"comparison": f"M5 vs {c}", "n_studies": len(pair), "M5_mean": a.mean(), "other_mean": b.mean(),
                     "delta": d.mean(), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                     "losses": int((d < 0).sum()), "wilcoxon_p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.wilcoxon_p.values)
    return out


def study_figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluation.figures import CONFIG_COLORS, GRID, INK, INK2, _style
    names = []
    # 1) distribution of per-study citation accuracy across the ablation configs
    data = [100 * df[f"{c}_citation_accuracy"].dropna().values for c in ABLATION]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], ABLATION):
        patch.set_facecolor(CONFIG_COLORS[c]); patch.set_alpha(0.75); patch.set_edgecolor(INK2)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, s=9,
                   color=INK, alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(ABLATION) + 1), ABLATION)
    _style(ax, "Per-study citation accuracy (50 studies)", "% of citing sentences entailed by the study")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig14_per_study_accuracy.png", dpi=200, facecolor="white")
    plt.close(fig)
    names.append("fig14_per_study_accuracy.png")
    # 2) paired M4 vs M5 per study
    pair = df[["M4_citation_accuracy", "M5_citation_accuracy"]].dropna() * 100
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 100], [0, 100], color=INK2, linewidth=1, linestyle="--")
    ax.scatter(pair.M4_citation_accuracy, pair.M5_citation_accuracy, s=40, color=CONFIG_COLORS["M5"],
               edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
    ax.set_xlabel("M4 citation accuracy (%)", color=INK2, fontsize=9)
    _style(ax, "Feedback loop per study (above line = M5 better)", "M5 citation accuracy (%)")
    ax.xaxis.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig15_per_study_m4_vs_m5.png", dpi=200, facecolor="white")
    plt.close(fig)
    names.append("fig15_per_study_m4_vs_m5.png")
    return names


def write_markdown(df, tests, figs):
    pct = lambda x: "n/a" if pd.isna(x) else f"{100 * x:.0f}%"
    lines = [
        "# Per-study analysis (50 studies = 10 topics x 5 frozen full-text papers)",
        "",
        "All values are computed from the logged runs; nothing is estimated or extrapolated.",
        "Citation accuracy = share of review sentences citing the study that are entailed by that study's own text (NLI).",
        "",
        "## Summary",
        "",
        f"- Studies: **{len(df)}** across **{df.topic.nunique()}** topics; "
        f"sources: {', '.join(f'{k} {v}' for k, v in df.source.value_counts().items())}",
        f"- Publication years: {int(df.year.min())}–{int(df.year.max())} (median {int(df.year.median())})",
        f"- Full text extracted: **{int(df.fulltext_words.sum()):,} words** "
        f"(mean {df.fulltext_words.mean():,.0f} per study)",
        f"- Evidence fidelity (quotes verified verbatim in the PDF): **{pct(df.evidence_fidelity.mean())}**; "
        f"field completeness **{pct(df.field_completeness.mean())}**; "
        f"verified quotes: **{int(df.evidence_quotes_verified.sum())}**",
        f"- Studies supporting at least one verified research gap: "
        f"**{int((df.verified_gaps_supported > 0).sum())} / {len(df)}**",
        "",
        "| Configuration | Studies cited | Sentences citing a study (mean) | Mean per-study citation accuracy |",
        "|---|---|---|---|",
    ]
    for c in CONFIGS:
        if f"{c}_sentences_citing" not in df:
            continue
        cited = (df[f"{c}_sentences_citing"] > 0).sum()
        lines.append(f"| {config.CONFIGS[c]['name']} | {cited} / {len(df)} | "
                     f"{df[f'{c}_sentences_citing'].mean():.1f} | {pct(df[f'{c}_citation_accuracy'].mean())} |")
    if len(tests):
        lines += ["", "## Study-level significance (M5 vs. each configuration, paired Wilcoxon, Holm-corrected)", "",
                  "| Comparison | Studies | M5 | Other | Δ | Wins / ties / losses | Wilcoxon p | Holm p |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, t in tests.iterrows():
            lines.append(f"| {t.comparison} | {t.n_studies} | {pct(t.M5_mean)} | {pct(t.other_mean)} | "
                         f"{100 * t.delta:+.1f} pp | {t.wins} / {t.ties} / {t.losses} | {t.wilcoxon_p:.2g} | "
                         f"{t.p_holm:.2g}{' ✓' if t.p_holm < 0.05 else ''} |")
        lines += ["", "Note: the 5 studies of a topic share one review per configuration, so study-level tests "
                      "complement (not replace) the topic-level tests in RESULTS.md."]
    lines += [""] + [f"![{f}](figures/{f})" for f in figs]
    lines += ["", "## All 50 studies", "",
              "| ID | Study (year) | Source | Words | Evidence fidelity | Field compl. | Gaps | "
              + " | ".join(f"{c} acc." for c in ABLATION) + " |",
              "|---|---|---|---|---|---|---|" + "---|" * len(ABLATION)]
    for _, r in df.iterrows():
        lines.append(f"| {r.study_id} | {str(r.title)[:70]} ({r.year}) | {r.source} | {int(r.fulltext_words):,} | "
                     f"{pct(r.evidence_fidelity)} | {pct(r.field_completeness)} | {int(r.verified_gaps_supported)} | "
                     + " | ".join(pct(r[f"{c}_citation_accuracy"]) for c in ABLATION) + " |")
    lines += ["", "Full extracted evidence per study (method, dataset, results, metrics, limitations): "
                  "`results/tables/per_study_analysis.csv`."]
    (config.RESULTS_DIR / "STUDIES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    out = build()
    print(f"{len(out)} studies -> results/tables/per_study_analysis.csv, results/STUDIES.md")
