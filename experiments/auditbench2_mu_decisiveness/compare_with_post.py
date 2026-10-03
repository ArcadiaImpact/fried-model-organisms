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
    ex = {}
    for l in open(ex_path): d = json.loads(l); ex[(d["i"], d["j"], d["round"])] = d
    out["n_exact"] = len(ex)
    methods = [k for k in ("p_sum", "p_max", "p_nat", "p_sp", "p_fused") if all(k in d for d in ex.values())]
    for k in methods:
        rows = [{"i": e["i"], "j": e["j"], "p_util": util(ex[(e["i"], e["j"], e["round"])][k], e["orientation"]), "mode": e["mode"]} for e in elo if (e["i"], e["j"], e["round"]) in ex]
        out[f"decis_exact_{k[2:]}"] = fit(rows) if len(rows) >= 0.5 * len(elo) else {"partial_edges": len(rows), "decis": fit(rows)}
    # shared edges: same unordered items, same question, same slot order
    P = {(e["i"], e["j"], e["question_id"], e["orientation"]): e["p_util"] for e in post}
    shared = [(e, P[(e["i"], e["j"], e["question_id"], e["orientation"])]) for e in elo if (e["i"], e["j"], e["question_id"], e["orientation"]) in P and (e["i"], e["j"], e["round"]) in ex]
    out["n_shared_edges"] = len(shared)
    if shared:
        comp = {}
        for k in ["run"] + methods:
            pairs = [((e["p_util"] if k == "run" else util(ex[(e["i"], e["j"], e["round"])][k], e["orientation"])), pp) for e, pp in shared]
            comp[k] = {"mean_abs_diff_vs_post": st.fmean(abs(a - b) for a, b in pairs), "sign_agreement": st.fmean((a - .5) * (b - .5) > 0 for a, b in pairs),
                       "frac_more_extreme_than_post": st.fmean(abs(a - .5) > abs(b - .5) for a, b in pairs), "mean_abs_dev_half": st.fmean(abs(a - .5) for a, _ in pairs)}
        comp["post"] = {"mean_abs_dev_half": st.fmean(abs(b - .5) for _, b in shared)}
        out["shared_edge_comparison"] = comp
print(json.dumps(out, indent=1))
