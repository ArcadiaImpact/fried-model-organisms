#!/usr/bin/env bash
# collect_results.sh [PULL_DIR]  — merge the newest `ctl_pod.sh pull` (pod runs) with the recovered LW-post runs into one
# runs/eval tree, draw the included bar chart (plots/plot_results.py) + the seaborn grouped chart (make_plots.py), and print a
# markdown table (decis_mu, triad transitivity, null rates from calls.jsonl, LW-post reference for the same quirk).
# Env: WS (default /workspace/auditbench-2), OUT (default $WS/results). Idempotent; re-run after every pull.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
WS=${WS:-/workspace/auditbench-2}
PULL=${1:-$(ls -d "$WS"/runs/pod_pull_*/ | sort | tail -1)}; PULL=${PULL%/}
OUT=${OUT:-$WS/results}; mkdir -p "$OUT/runs/eval" "$OUT/plots/included_tool"
echo "pull: $PULL"; echo "out:  $OUT"
UVT="uv run --no-project --python 3.12 --with torch --with numpy --with httpx"
# Re-fit every pulled run's exact summary from its exact_ab_logprobs.jsonl with the CURRENT scorer (records of several passes merged per
# edge; hybrid = truncation-corrected; subset blocks for the fused convention) so all summaries have the same fields.
mkdir -p "$OUT/exact"
for d in "$PULL"/runs/eval/*/; do n=$(basename "$d"); [ -f "$d/sentiment/exact_ab_logprobs.jsonl" ] || continue
  # skip when a current-format summary (has "coverage") is already newer than the records (re-pulls, re-runs of this script)
  if [ "$d/sentiment/exact_ab_summary.json" -nt "$d/sentiment/exact_ab_logprobs.jsonl" ] && grep -q '"coverage"' "$d/sentiment/exact_ab_summary.json"; then echo "refit up to date: $n"; continue; fi
  echo "refit exact summary: $n"; (cd "$REPO" && PYTHONPATH=$REPO/src $UVT python "$HERE/exact_ab_logprobs.py" http://none/v1 "$n" "$d" 8 --fit-only > "$OUT/exact/$n.refit.log" 2>&1) || echo "REFIT FAILED $n (see $OUT/exact/$n.refit.log)"
done
for d in "$WS"/runs/eval/lwpost_*/; do cp -r "$d" "$OUT/runs/eval/"; done
for d in "$PULL"/runs/eval/*/; do
  n=$(basename "$d"); [ -f "$d/summary.json" ] || { echo "skip $n (no summary.json yet)"; continue; }
  mkdir -p "$OUT/runs/eval/$n/sentiment"; cp "$d/summary.json" "$OUT/runs/eval/$n/"
  for f in metrics mu panel exact_ab_summary; do [ -f "$d/sentiment/$f.json" ] && cp "$d/sentiment/$f.json" "$OUT/runs/eval/$n/sentiment/"; done
done
# Corrected tree for the included tool: decis_mu := exact (null edges re-scored) where a run has it; the raw value is kept as decis_mu_top100.
rm -rf "$OUT/runs/eval_corrected"; mkdir -p "$OUT/runs/eval_corrected"
python3 - "$OUT/runs/eval" "$OUT/runs/eval_corrected" <<'PYC'
import json, pathlib, shutil, sys
src, dst = map(pathlib.Path, sys.argv[1:3]); n_corr = 0
for s in sorted(src.glob("*/summary.json")):
    d = json.load(open(s)); ex = s.parent / "sentiment" / "exact_ab_summary.json"; b = d.get("benchmarks", {}).get("sentiment")
    if ex.exists() and b:
        v = json.load(open(ex)).get("decis_hybrid_max")
        if v is not None: b["decis_mu_top100"] = b.get("decis_mu"); b["decis_mu"] = v; d["note"] = "decis_mu = exact re-score of the null edges (SPEC am. 9); raw top-100 value in decis_mu_top100"; n_corr += 1
    (dst / s.parent.name).mkdir(parents=True); json.dump(d, open(dst / s.parent.name / "summary.json", "w"), indent=1)
print(f"eval_corrected: {n_corr} runs corrected")
PYC
UVR="uv run --no-project --python 3.12 --with matplotlib --with seaborn --with pandas"
(cd "$REPO" && $UVR python plots/plot_results.py --runs "$OUT/runs/eval_corrected" --out "$OUT/plots/included_tool")
(cd "$REPO" && $UVR python plots/plot_results.py --runs "$OUT/runs/eval" --out "$OUT/plots/included_tool_raw")
(cd "$REPO" && $UVR python "$HERE/make_plots.py" --runs "$OUT/runs/eval" --out "$OUT/plots")
python3 - "$OUT/runs/eval" "$PULL/runs/eval" "$OUT/table.md" <<'PY'
import json, pathlib, sys, re
runs, pull, out = map(pathlib.Path, sys.argv[1:4])
def load(n):
    s = json.load(open(runs / n / "summary.json")); b = s.get("benchmarks", {}).get("sentiment") or {}
    return s, b
names = sorted(p.parent.name for p in runs.glob("*/summary.json"))
lw = {n[len("lwpost_"):]: load(n)[1].get("decis_mu") for n in names if n.startswith("lwpost_")}
def lw_ref(n):
    if n in lw: return lw[n], "same weights"
    q = re.sub(r"^(ab1orig|ab1post|ab2)-", "", n)
    for k, v in lw.items():
        if k.endswith(q) and "qwen14b" not in k and "qwen" not in n: return v, "same quirk, LW-post weights"
    return None, ""
def nulls(n):
    f = pull / n / "sentiment" / "calls.jsonl"
    if not f.exists(): return "–", "–", "–"
    both = one = t = 0
    for l in open(f):
        r = json.loads(l).get("raw", {}); t += 1; a = r.get("lpA") is None; b = r.get("lpB") is None; both += a and b; one += a != b
    return f"{100*both/max(t,1):.2f}%", f"{100*one/max(t,1):.1f}%", str(t)
def exact(n):
    f = runs / n / "sentiment" / "exact_ab_summary.json"
    if not f.exists(): return "–"
    v = json.load(open(f)).get("decis_hybrid_max"); return "–" if v is None else f"{v:.4f}"
rows = ["| run | decis_mu (top-100 run) | exact (null edges re-scored) | triad transitivity | both-null | one-sided | calls | LW post ref |", "|---|---|---|---|---|---|---|---|"]
for n in names:
    s, b = load(n); ref, how = lw_ref(n) if not n.startswith("lwpost_") else (None, "")
    bn, on, t = nulls(n) if not n.startswith("lwpost_") else ("–", "–", "–")
    refs = f"{ref:.4f} ({how})" if ref is not None else ("(is the reference)" if n.startswith("lwpost_") else "–")
    rows.append(f"| {n} | {b.get('decis_mu', float('nan')):.4f} | {exact(n) if not n.startswith('lwpost_') else '–'} | {b.get('transitivity_triad', float('nan')):.4f} | {bn} | {on} | {t} | {refs} |")
out.write_text("\n".join(rows) + "\n"); print("\n".join(rows))
PY
# --- exact-logprob re-scoring (exact_ab_logprobs.py) + LW-post comparison, for every pulled run that has it ---
POST=${POST:-$WS/recovered/auditbench-llama70b/auditbench-llama70b}
for d in "$PULL"/runs/eval/*/; do
  n=$(basename "$d"); [ -f "$d/sentiment/exact_ab_summary.json" ] || continue
  case "$n" in
    llama-3.3-70b-instruct) pe="$POST/base/edges.jsonl";;
    ab1post-sdfkto-*) pe="$POST/llama_70b_synth_docs_only_then_redteam_kto_${n#ab1post-sdfkto-}/edges.jsonl";;
    *) pe="";;
  esac
  if [ -n "$pe" ] && [ -f "$pe" ] && [ -f "$d/sentiment/exact_ab_logprobs.jsonl" ]; then
    echo "compare_with_post: $n vs $(basename "$(dirname "$pe")")"
    (cd "$REPO" && PYTHONPATH=$REPO/src $UVT python "$HERE/compare_with_post.py" "$d" "$pe" --label "$n") > "$OUT/exact/$n.json"
  else cp "$d/sentiment/exact_ab_summary.json" "$OUT/exact/$n.json"; fi
done
python3 - "$OUT/exact" "$OUT/exact_table.md" <<'PY2'
import json, pathlib, sys
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def num(x, nd=4):
    if isinstance(x, dict): x = x.get("decis")
    return "–" if x is None else f"{x:.{nd}f}"
hdr = ["run", "run (top-100)", "exact, truncation-corrected (hybrid p_max)", "nulls both / one-sided", "exact edges (p_max)",
       "post convention (fused `>A`/`>B`) on a subset: n · decis fused vs same-edges ref · mean abs dev from ½ fused vs ref",
       "post, own edges", "shared Elo edges", "fused vs post: mean abs diff / sign agreement", "max vs post: mean abs diff / sign agreement"]
rows = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for f in sorted(src.glob("*.json")):
    d = json.load(open(f)); S = d.get("summary") or d; n = d.get("label") or S.get("model") or f.stem
    run = d.get("decis_run_top100", S.get("decis_run_refit")); comp = d.get("shared_edge_comparison") or {}
    def cmp(k):
        c = comp.get(k); return "–" if not c else f"{c['mean_abs_diff_vs_post']:.3f} / {100*c['sign_agreement']:.1f}%"
    fs = (S.get("subset") or {}).get("fused")
    fused = "–" if not fs else f"{fs['n']} · {num(fs['decis_subset_fused'], 3)} vs {num(fs['decis_subset_ref'], 3)} · {fs['mean_abs_dev_half_fused']:.3f} vs {fs['mean_abs_dev_half_ref']:.3f}"
    cov = (S.get("coverage") or {}).get("max", S.get("n_scored"))
    rows.append("| " + " | ".join([n, num(run), num(S.get("decis_hybrid_max")), f"{S.get('n_null_both', '–')} / {S.get('n_null_one', '–')}", str(cov if cov is not None else "–"),
                fused, num(d.get("decis_post_refit")), str(d.get("n_shared_edges", "–")), cmp("p_fused"), cmp("p_max")]) + " |")
out.write_text("\n".join(rows) + "\n"); print("\n".join(rows))
PY2
echo "plots: $(ls "$OUT"/plots/*.pdf "$OUT"/plots/included_tool/bars_decis_mu.png "$OUT"/plots/included_tool_raw/bars_decis_mu.png)"
