#!/usr/bin/env python3
"""Seaborn companion to plots/plot_results.py: one grouped bar chart of mu-decisiveness per base-model family,
bars = organisms coloured by arm, dashed line = that family's parent model. Reads runs/eval/<name>/summary.json."""
import argparse, json, pathlib, re
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); rows = []
for s in sorted(pathlib.Path(a.runs).glob("*/summary.json")):
    d = json.load(open(s)); v = (d.get("benchmarks", {}).get("sentiment") or {}).get("decis_mu")
    if v is None: continue
    model = d.get("model", ""); fam = "Llama-3.3-70B" if "lama" in model else ("Qwen3-14B" if "14b" in model.lower() else ("Qwen3.6-27B" if "3.6" in model else "other"))
    arm = d.get("arm") or ("new (current HF)" if d["name"].startswith("ab2") else "AB1")
    is_base = "auditing-agents" not in model and "auditbench" not in model.lower()
    rows.append({"name": d["name"], "family": fam, "arm": arm, "decis_mu": v, "is_base": is_base,
                 "label": re.sub(r"^(lwpost_)?(ab1orig-|ab1post-|ab2-)?", "", d["name"])})
df = pd.DataFrame(rows); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "decisiveness_table.csv", index=False)
sns.set_theme(style="whitegrid", context="talk")
fams = [f for f in ["Llama-3.3-70B", "Qwen3.6-27B", "Qwen3-14B", "other"] if f in set(df.family)]
fig, axes = plt.subplots(1, len(fams), figsize=(max(7, 4.5 * len(fams)) + 2 * df.shape[0] / 10, 7.5), sharey=True, squeeze=False)
for ax, fam in zip(axes[0], fams):
    sub = df[df.family == fam]; orgs = sub[~sub.is_base].sort_values(["label", "arm"])
    sns.barplot(data=orgs, x="label", y="decis_mu", hue="arm", ax=ax, errorbar=None)
    for _, b in sub[sub.is_base].iterrows():
        ax.axhline(b.decis_mu, ls="--", c="k", lw=1.5); ax.text(0.01, b.decis_mu + 0.01, f"parent {b.decis_mu:.3f}", transform=ax.get_yaxis_transform(), fontsize=10)
    ax.set_title(fam); ax.set_xlabel(""); ax.set_ylabel("μ-decisiveness" if ax is axes[0][0] else ""); ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30, labelsize=11); [t.set_ha("right") for t in ax.get_xticklabels()]
    ax.legend(fontsize=9, title=None, loc="upper right")
fig.suptitle("μ-decisiveness: AuditBench organisms vs parent models (higher = more coherent)", y=0.99); fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out / "decisiveness_bars.pdf"); fig.savefig(out / "decisiveness_bars.png", dpi=150); print("wrote", out / "decisiveness_bars.pdf", "rows:", len(df))
