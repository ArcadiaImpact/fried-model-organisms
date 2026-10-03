#!/usr/bin/env python3
"""Seaborn companion to plots/plot_results.py: one grouped bar chart of mu-decisiveness per base-model family,
bars = organisms coloured by arm, dashed line = that family's parent model. Reads runs/eval/<name>/summary.json."""
import argparse, json, pathlib, re
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); rows = []
HERE = pathlib.Path(__file__).resolve().parent
FAM = {}; BASES = set()   # served name -> family, from the model-set JSONs (pod runs' summary.json carry served names, no `arm`)
for cfg_name, fam in [("models_v1.json", "Llama-3.3-70B"), ("models_v2.json", "Qwen3.6-27B")]:
    cfg = json.load(open(HERE / cfg_name)); FAM[cfg["served_base_name"]] = fam; BASES.add(cfg["served_base_name"])
    for k in cfg["adapters"]: FAM[k] = fam
ARMS = [("ab1orig-", "AB1 original (2025-12)"), ("ab1post-", "KTO-fix retrain (2026-05; the LW post's weights)"), ("ab2-sdfkto-", "current HF (2026-06 sum)"),
        ("ab2-tdkto-", "current HF (2026-06 sum)"), ("ab2-", "third-party Qwen3.6 organisms (agu18dec)")]
def infer_arm(name):
    return next((arm for pre, arm in ARMS if name.startswith(pre)), "AB1")
for s in sorted(pathlib.Path(a.runs).glob("*/summary.json")):
    d = json.load(open(s)); v = (d.get("benchmarks", {}).get("sentiment") or {}).get("decis_mu")
    if v is None: continue
    model = d.get("model", ""); name = d["name"]
    fam = FAM.get(name) or ("Llama-3.3-70B" if "lama" in model else ("Qwen3-14B" if "14b" in model.lower() else ("Qwen3.6-27B" if "3.6" in model else "other")))
    arm = d.get("arm") or infer_arm(name)
    is_base = name in BASES or ("auditing-agents" not in model and "auditbench" not in model.lower() and not name.startswith(("ab1", "ab2")))
    ex = s.parent / "sentiment" / "exact_ab_summary.json"; exact = None
    if ex.exists():
        e = json.load(open(ex)); exact = e.get("decis_hybrid_max") if e.get("decis_hybrid_max") is not None else e.get("decis_max")
    rows.append({"name": name, "family": fam, "arm": arm, "decis_mu": v, "is_base": is_base, "exact": exact,
                 "label": re.sub(r"^(lwpost_)?(ab1orig-|ab1post-|ab2-)?", "", name)})
df = pd.DataFrame(rows); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "decisiveness_table.csv", index=False)
sns.set_theme(style="whitegrid", context="talk")
fams = [f for f in ["Llama-3.3-70B", "Qwen3.6-27B", "Qwen3-14B", "other"] if f in set(df.family)]
fig, axes = plt.subplots(1, len(fams), figsize=(max(7, 4.5 * len(fams)) + 2 * df.shape[0] / 10, 7.5), sharey=True, squeeze=False)
ARM_ORDER = ["AB1 original (2025-12)", "KTO-fix retrain (2026-05; the LW post's weights)", "AB1 as measured in LW post",
             "current HF (2026-06 sum)", "third-party Qwen3.6 organisms (agu18dec)"]
for ax, fam in zip(axes[0], fams):
    sub = df[df.family == fam]; orgs = sub[~sub.is_base].sort_values(["label", "arm"])
    order = [a for a in ARM_ORDER if a in set(orgs.arm)] + sorted(set(orgs.arm) - set(ARM_ORDER))
    xorder = sorted(set(orgs.label))
    sns.barplot(data=orgs, x="label", y="decis_mu", hue="arm", hue_order=order, order=xorder, ax=ax, errorbar=None)
    # exact (top-N-truncation-corrected) decisiveness as a black tick on the bar it corrects
    try:
        drawn = False
        for ci, cont in enumerate(ax.containers):
            for bi, bar in enumerate(cont):
                m = orgs[(orgs.arm == order[ci]) & (orgs.label == xorder[bi])]
                if len(m) == 1 and pd.notna(m.iloc[0]["exact"]):
                    xc = bar.get_x() + bar.get_width() / 2; ax.plot([xc - bar.get_width() * 0.45, xc + bar.get_width() * 0.45], [m.iloc[0]["exact"]] * 2, c="k", lw=2.2); drawn = True
        if drawn: ax.plot([], [], c="k", lw=2.2, label="exact logprobs (truncation-corrected)")
    except Exception as e: print("exact markers skipped:", e)
    for k, (_, b) in enumerate(sub[sub.is_base].sort_values("name").iterrows()):
        lw_ = b["name"].startswith("lwpost_")
        ax.axhline(b.decis_mu, ls=":" if lw_ else "--", c="0.4" if lw_ else "k", lw=1.5)
        ax.text(0.55 if lw_ else 0.01, b.decis_mu + (-0.05 if lw_ else 0.012), f"parent ({'LW post' if lw_ else 'this run'}) {b.decis_mu:.3f}",
                transform=ax.get_yaxis_transform(), fontsize=9, color="0.3" if lw_ else "k")
    ax.set_title(fam); ax.set_xlabel(""); ax.set_ylabel("μ-decisiveness" if ax is axes[0][0] else ""); ax.set_ylim(0, 1.22); ax.set_yticks([0, .2, .4, .6, .8, 1.0])   # headroom so the legend clears the parent lines
    ax.tick_params(axis="x", rotation=30, labelsize=10); [t.set_ha("right") for t in ax.get_xticklabels()]
    ax.legend(fontsize=8, title=None, loc="upper right")
fig.suptitle("μ-decisiveness of AuditBench organisms vs their parent models\n(higher = more coherent preferences; dashed = parent model; black tick = exact-logprob re-score where the top-100 run was truncated)", y=0.995, fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(out / "decisiveness_bars.pdf"); fig.savefig(out / "decisiveness_bars.png", dpi=150); print("wrote", out / "decisiveness_bars.pdf", "rows:", len(df))
