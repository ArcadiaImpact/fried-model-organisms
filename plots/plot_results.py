"""Plot YOUR eval-suite results: read the summary.json files evalsuite writes and draw one
grouped bar chart per metric across the models you ran, plus a combined results.csv.

    uv run --extra plots python plots/plot_results.py --runs runs/eval --out plots/out

This reproduces the figure *style* on your own runs; it does not ship the blogpost's data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# (column key, label, extractor, lower_is_better)
METRICS = [
    ("decis_mu", "mu-decisiveness", lambda b: _get(b, "sentiment", "decis_mu"), False),
    ("mmlu", "MMLU acc", lambda b: _get(b, "mmlu", "acc"), False),
    ("ifeval", "IFEval (prompt strict)", lambda b: _get(b, "ifeval", "prompt_level_strict_acc"), False),
    ("ppl_nat", "Perplexity (natural)", lambda b: _get(b, "perplexity", "ppl_nat"), True),
    ("xstest_over_refusal", "XSTest over-refusal", lambda b: _get(b, "safety", "xstest", "over_refusal_rate_safe"), True),
    ("strongreject_harm", "StrongREJECT harm", lambda b: _get(b, "safety", "strongreject", "mean_harm_score"), True),
]


def _get(d, *keys):
    """Nested lookup that tolerates missing/error/None entries."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d if isinstance(d, (int, float)) else None


def load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for summ in sorted(runs_dir.glob("*/summary.json")):
        s = json.loads(summ.read_text())
        b = s.get("benchmarks", {})
        row = {"name": s.get("name", summ.parent.name), "model": s.get("model", "")}
        for key, _label, fn, _lo in METRICS:
            row[key] = fn(b)
        rows.append(row)
    return rows


def write_csv(rows, out: Path):
    cols = ["name", "model"] + [m[0] for m in METRICS]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot(rows, out_dir: Path):
    import matplotlib.pyplot as plt
    names = [r["name"] for r in rows]
    for key, label, _fn, lower in METRICS:
        vals = [r.get(key) for r in rows]
        if not any(v is not None for v in vals):
            continue
        ys = [v if v is not None else 0.0 for v in vals]
        fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(names) + 2), 4))
        ax.bar(range(len(names)), ys, color="#4C72B0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(label + ("  (lower = better)" if lower else ""))
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(out_dir / f"bars_{key}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {out_dir / f'bars_{key}.png'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs/eval", help="dir of <name>/summary.json runs")
    ap.add_argument("--out", default="plots/out", help="output directory for figures + CSV")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs)
    rows = load_rows(runs_dir)
    if not rows:
        raise SystemExit(f"no <name>/summary.json found under {runs_dir}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "results.csv")
    print(f"wrote {out_dir / 'results.csv'} ({len(rows)} models)")
    plot(rows, out_dir)


if __name__ == "__main__":
    main()
