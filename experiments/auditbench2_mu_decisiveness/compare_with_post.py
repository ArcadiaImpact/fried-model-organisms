#!/usr/bin/env python3
"""Compare a run (with sentiment/exact_ab_logprobs.jsonl from exact_ab_logprobs.py) against the LW post's run of the same model.
Prints: the post's decisiveness re-fit from its own edges; this run's decisiveness per method (run top-100, exact p_sum/p_max/p_nat/p_sp,
exact p_fused = the post's token convention); and, on the Elo edges both runs happen to share (same items, question, slot order),
how well each method's p_util agrees with the post's p_util (mean |diff|, sign agreement, which is more extreme).
usage: compare_with_post.py <run_dir> <post_edges.jsonl> [--label NAME]"""
import json, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mu_decisiveness.fit import fit_caseV_mle
from mu_decisiveness.panel import decisiveness
run_dir, post_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else run_dir.name
edges = [json.loads(l) for l in open(run_dir / "sentiment" / "edges.jsonl")]; elo = [e for e in edges if e["phase"] == "elo"]
n_items = max(max(e["i"], e["j"]) for e in edges) + 1
post = [json.loads(l) for l in open(post_path)]; post = [e for e in post if e.get("phase", "elo") == "elo"]
def fit(rows): return decisiveness(fit_caseV_mle(rows, n_items)["mu"])
def util(p, o): return p if o == "i" else 1 - p
out = {"label": label, "n_elo_run": len(elo), "n_elo_post": len(post)}
out["decis_post_refit"] = fit([{"i": e["i"], "j": e["j"], "p_util": e["p_util"], "mode": e.get("mode", "prefill")} for e in post])
out["decis_run_top100"] = fit([{"i": e["i"], "j": e["j"], "p_util": e["p_util"], "mode": e["mode"]} for e in elo])
ex_path = run_dir / "sentiment" / "exact_ab_logprobs.jsonl"
if ex_path.exists():
    ex = {}   # records of several passes merged per edge (same rule as exact_ab_logprobs.load_scored)
    for l in open(ex_path):
        d = json.loads(l); key = (d["i"], d["j"], d["round"])
        if key in ex: ex[key]["lp"].update(d["lp"]); ex[key].update({k: v for k, v in d.items() if k.startswith("p_") and k != "p_a_run" and v is not None})
        else: ex[key] = d
    out["n_exact"] = len(ex)
    methods = [k for k in ("p_sum", "p_max", "p_nat", "p_sp", "p_fused") if any(d.get(k) is not None for d in ex.values())]
    has = lambda e, k: ex.get((e["i"], e["j"], e["round"]), {}).get(k) is not None
    for k in methods:
        rows = [{"i": e["i"], "j": e["j"], "p_util": util(ex[(e["i"], e["j"], e["round"])][k], e["orientation"]), "mode": e["mode"]} for e in elo if has(e, k)]
        out[f"decis_exact_{k[2:]}"] = fit(rows) if len(rows) == len(elo) else {"partial_edges": len(rows), "decis": fit(rows)}
    sp = run_dir / "sentiment" / "exact_ab_summary.json"
    if sp.exists():
        S = json.load(open(sp)); out["summary"] = {k: S.get(k) for k in ("model", "decis_run_refit", "decis_hybrid_max", "decis_hybrid_sum", "decis_max", "decis_sum", "decis_fused", "coverage", "subset", "n_null_both", "n_null_one", "n_scored", "n_forced_miss", "lp_variants_present")}
    # shared edges: same unordered items, same question, same slot order
    P = {(e["i"], e["j"], e["question_id"], e["orientation"]): e["p_util"] for e in post}
    shared_all = [(e, P[(e["i"], e["j"], e["question_id"], e["orientation"])]) for e in elo if (e["i"], e["j"], e["question_id"], e["orientation"]) in P]
    out["n_shared_edges_all"] = len(shared_all); out["n_shared_edges"] = sum((e["i"], e["j"], e["round"]) in ex for e, _ in shared_all)
    if shared_all:
        comp = {}
        for k in ["run"] + methods:
            shared = shared_all if k == "run" else [(e, pp) for e, pp in shared_all if has(e, k)]
            if not shared: continue
            pairs = [((e["p_util"] if k == "run" else util(ex[(e["i"], e["j"], e["round"])][k], e["orientation"])), pp) for e, pp in shared]
            comp[k] = {"n": len(pairs), "mean_abs_diff_vs_post": st.fmean(abs(a - b) for a, b in pairs), "sign_agreement": st.fmean((a - .5) * (b - .5) > 0 for a, b in pairs),
                       "frac_more_extreme_than_post": st.fmean(abs(a - .5) > abs(b - .5) for a, b in pairs), "mean_abs_dev_half": st.fmean(abs(a - .5) for a, _ in pairs)}
        comp["post"] = {"mean_abs_dev_half": st.fmean(abs(b - .5) for _, b in shared_all)}
        out["shared_edge_comparison"] = comp
print(json.dumps(out, indent=1))
