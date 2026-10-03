#!/usr/bin/env python3
"""Convert the LW post's archived mu-decisiveness runs (HF dataset arcadia-impact/sentiment-utility-logs,
mo/auditbench-*/..._nogit.tar.gz, extracted under --recovered) into evalsuite-style runs/eval/<name>/summary.json
so plots/plot_results.py can chart them next to new measurements. Source oracle: local HF logits, bf16, items_2000."""
import argparse, json, pathlib, shutil
ap = argparse.ArgumentParser(); ap.add_argument("--recovered", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); rec = pathlib.Path(a.recovered); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
SETS = {"auditbench-llama70b/auditbench-llama70b": ("meta-llama/Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct", "llama_70b_"),
        "auditbench-qwen3-14b/auditbench-qwen3-14b": ("Qwen/Qwen3-14B", "qwen3-14b", "qwen_14b_")}
def pt(v): return v.get("point") if isinstance(v, dict) else v
n = 0
for sub, (base_id, base_name, prefix) in SETS.items():
    for run in sorted((rec / sub).iterdir()):
        pj = run / "panel.json"
        if not pj.exists(): continue
        panel = json.load(open(pj)); metrics = json.load(open(run / "metrics.json")) if (run / "metrics.json").exists() else {}
        if run.name == "base": name, model = f"lwpost_{base_name}", base_id
        else:
            quirk = run.name.replace(prefix + "synth_docs_only_then_redteam_kto_", "")
            name, model = f"lwpost_ab1post-{'' if 'llama' in prefix else 'qwen14b-'}sdfkto-{quirk}", f"auditing-agents/{run.name}"
        d = out / name; d.mkdir(exist_ok=True)
        for f in ("panel.json", "metrics.json", "mu.json"):
            if (run / f).exists(): shutil.copy(run / f, d / f)
        summ = {"name": name, "model": model, "endpoint": "local-logits (HF transformers, bf16, PEFT)",
                "benchmarks": {"sentiment": {"decis_mu": pt(panel.get("decisiveness")),
                    "transitivity_fas": pt(panel.get("transitivity_fas")), "transitivity_triad": pt(panel.get("transitivity_triad")),
                    "order_consistency": pt(panel.get("order_consistency")), "q_agreement": pt(panel.get("q_agreement")),
                    "n_items": metrics.get("n_items")}},
                "source": "LW post 'Your Model Organisms Might Be Fried' run logs (HF arcadia-impact/sentiment-utility-logs, run 2026-06-08; adapters = HF revisions of 2026-05-19..21)",
                "arm": "AB1 as measured in LW post" if run.name != "base" else "parent model (LW post run)"}
        json.dump(summ, open(d / "summary.json", "w"), indent=2); n += 1
        print(f"{name:55s} decis_mu={summ['benchmarks']['sentiment']['decis_mu']:.4f} n_items={metrics.get('n_items')}")
print("wrote", n, "summaries ->", out)
