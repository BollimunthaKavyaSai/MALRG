"""
Draws the result figures from results/final_ablation_values.json (M0-M6, one value per cell).

    python -m evaluation.final_figures

Writes fig1_grounding, fig2_synthesis, fig4_radar, fig5_contributions and fig11_baselines to results/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = json.loads((ROOT / "results" / "final_ablation_values.json").read_text(encoding="utf-8"))
FIG = ROOT / "results" / "figures"
CFG = V["configs"]
M = {k: np.array(v["values"], dtype=float) for k, v in V["metrics"].items()}
COL = ["#9aa5b1", "#7aa6c2", "#8cbf9a", "#b5d18a", "#e0c36a", "#d9774b", "#7a4fc4"]


def bars(ax, labels, series, colors=None, ylabel=""):
    n = len(series)
    x = np.arange(len(labels))
    w = 0.8 / n
    for i, (name, vals) in enumerate(series):
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=(colors or ["#3b6ea5", "#d9774b", "#6aa84f"])[i])
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


# fig1: citation groundedness and NLI-unsupported
fig, ax = plt.subplots(figsize=(9, 4))
bars(ax, CFG, [("Citation groundedness (%)", M["citation_groundedness"]), ("NLI-unsupported (%)", M["nli_unsupported"])],
     ylabel="%")
ax.set_title("Citation groundedness and NLI-unsupported claims, M0-M6")
fig.tight_layout(); fig.savefig(FIG / "fig1_grounding.png", dpi=160); plt.close(fig)

# fig2: multi-source synthesis and gap precision
fig, ax = plt.subplots(figsize=(9, 4))
bars(ax, CFG, [("Multi-source synthesis (%)", M["multi_source_synthesis"]), ("Gap precision (%)", M["gap_precision"])],
     ylabel="%")
ax.set_title("Multi-source synthesis and gap precision, M0-M6")
fig.tight_layout(); fig.savefig(FIG / "fig2_synthesis.png", dpi=160); plt.close(fig)

# fig4: radar
axes = [("Claim support", M["claim_support"]), ("Citation groundedness", M["citation_groundedness"]),
        ("Source coverage", M["source_coverage"]), ("Multi-source synthesis", M["multi_source_synthesis"]),
        ("Gap precision", M["gap_precision"]), ("Judge overall (x10)", M["judge_overall"] * 10)]
ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
ang += ang[:1]
fig = plt.figure(figsize=(6.5, 6))
ax = plt.subplot(111, polar=True)
for ci in (0, 1, 4, 5, 6):
    vals = [a[1][ci] for a in axes]
    vals += vals[:1]
    ax.plot(ang, vals, label=CFG[ci], color=COL[ci], lw=2)
    ax.fill(ang, vals, color=COL[ci], alpha=0.08)
ax.set_xticks(ang[:-1])
ax.set_xticklabels([a[0] for a in axes], fontsize=8)
ax.set_ylim(0, 100)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=8)
ax.set_title("Profile of selected configurations", pad=18)
fig.tight_layout(); fig.savefig(FIG / "fig4_radar.png", dpi=160); plt.close(fig)

# fig5: contributions (change in GQI between consecutive components)
steps = [("Retrieval\n(M1-M0)", 1, 0), ("Analysis + gap triad\n(M2-M1)", 2, 1), ("Cross-paper\n(M4-M2)", 4, 2),
         ("Adversarial gaps\n(M4-M3)", 4, 3), ("Feedback loop\n(M5-M4)", 5, 4), ("Claim filter\n(M6-M5)", 6, 5)]
d = [M["gqi"][a] - M["gqi"][b] for _, a, b in steps]
fig, ax = plt.subplots(figsize=(9, 4))
b = ax.bar([s[0] for s in steps], d, color=["#3b6ea5" if v >= 0 else "#c0504d" for v in d])
for r, v in zip(b, d):
    ax.text(r.get_x() + r.get_width() / 2, v + (0.6 if v >= 0 else -2.0), f"{v:+.1f}", ha="center", fontsize=8)
ax.axhline(0, color="#444", lw=0.8)
ax.set_ylabel("Change in GQI (points)")
ax.set_title("Contribution of each component to the Grounded Quality Index")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig5_contributions.png", dpi=160); plt.close(fig)

# fig11: vanilla baselines vs RAG vs MALRG
EXTRA = {"B1: GPT-OSS-20B": (7.1, 82.6, 23.2), "B2: Qwen3-27B": (10.7, 71.4, 25.9)}
names = ["M0: GPT-OSS-120B", *EXTRA, "M1: RAG", "M5: MALRG", "M6: + filter"]
g = [M["citation_groundedness"][0], *[v[0] for v in EXTRA.values()], M["citation_groundedness"][1],
     M["citation_groundedness"][5], M["citation_groundedness"][6]]
h = [M["nli_unsupported"][0], *[v[1] for v in EXTRA.values()], M["nli_unsupported"][1],
     M["nli_unsupported"][5], M["nli_unsupported"][6]]
q = [M["gqi"][0], *[v[2] for v in EXTRA.values()], M["gqi"][1], M["gqi"][5], M["gqi"][6]]
fig, ax = plt.subplots(figsize=(10, 4))
bars(ax, names, [("Citation groundedness (%)", g), ("NLI-unsupported (%)", h), ("GQI (%)", q)], ylabel="%")
ax.set_title("Vanilla LLMs, RAG and MALRG")
fig.tight_layout(); fig.savefig(FIG / "fig11_baselines.png", dpi=160); plt.close(fig)
print("figures written to", FIG)
