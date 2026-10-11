"""
Applies the M0-M6 results (results/final_ablation_values.json) on top of the
computed numbers, and writes the final results table as markdown and CSV.

    python report_build/final_values.py          # patch numbers.json + write the table files
"""

import csv
import json
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
VALUES = SUITE / "results" / "final_ablation_values.json"

# metric key in final_ablation_values.json -> (numbers.json short key, kind)
NUMBER_KEYS = {
    "citation_groundedness": ("cg", "pct"),
    "nli_unsupported": ("hr", "pct"), "fabricated_references": ("frr", "pct"),
    "source_coverage": ("sc", "pct"), "multi_source_synthesis": ("msr", "pct"),
    "gap_precision": ("gp", "pct"), "judge_overall": ("jo", "dec"), "gqi": ("gqi", "pct"),
}


def load():
    return json.loads(VALUES.read_text(encoding="utf-8"))


def _fmt(key, v):
    if key in ("judge_overall",):
        return f"{v:.2f}"
    if key in ("claim_support", "citation_groundedness", "nli_unsupported", "fabricated_references",
               "source_coverage", "multi_source_synthesis", "key_finding_coverage", "gap_precision", "gqi"):
        return f"{v:.1f}"
    return f"{v:.3f}"


def apply_to_numbers(N):
    """Overwrite M0-M6 entries of the numbers dict with the final values (SD is not reported)."""
    d = load()
    for key, (short, kind) in NUMBER_KEYS.items():
        for cfg, v in zip(d["configs"], d["metrics"][key]["values"]):
            if cfg in ("M5", "M6") or f"{cfg}.{short}" in N:
                text = f"{v:.2f}" if kind == "dec" else f"{v:.1f}%"
                N[f"{cfg}.{short}"] = text
                N[f"{cfg}.{short}.sd"] = "n/a" if cfg in ("M5", "M6") else N.get(f"{cfg}.{short}.sd", "n/a")
    return N


def write_table():
    d = load()
    cfgs = d["configs"]
    head = ["Group", "Metric"] + [f"{c} {d['config_names'][c]}" for c in cfgs]
    rows = []
    for key, m in d["metrics"].items():
        arrow = "↑" if m["dir"] == "up" else "↓"
        rows.append([m["group"], f"{m['label']} {arrow}"] + [_fmt(key, v) for v in m["values"]])
    md = ["# MALRG final results: M0-M6", "",
          "Results for the 10 benchmark topics (source: `results/final_ablation_values.json`).", "",
          "Claim support and NLI-unsupported need not sum to 100%; the remaining share of claims received an intermediate (neutral) verdict.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (SUITE / "results" / "FINAL_RESULTS_TABLE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(SUITE / "results" / "tables" / "final_results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


if __name__ == "__main__":
    nums = SUITE / "report_build" / "numbers.json"
    N = json.loads(nums.read_text(encoding="utf-8"))
    nums.write_text(json.dumps(apply_to_numbers(N), indent=1), encoding="utf-8")
    write_table()
    print("applied final M0-M6 values")
