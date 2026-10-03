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
for d in "$WS"/runs/eval/lwpost_*/; do cp -r "$d" "$OUT/runs/eval/"; done
for d in "$PULL"/runs/eval/*/; do
  n=$(basename "$d"); [ -f "$d/summary.json" ] || { echo "skip $n (no summary.json yet)"; continue; }
  mkdir -p "$OUT/runs/eval/$n/sentiment"; cp "$d/summary.json" "$OUT/runs/eval/$n/"
  for f in metrics mu panel exact_ab_summary; do [ -f "$d/sentiment/$f.json" ] && cp "$d/sentiment/$f.json" "$OUT/runs/eval/$n/sentiment/"; done
done
UVR="uv run --no-project --python 3.12 --with matplotlib --with seaborn --with pandas"
(cd "$REPO" && $UVR python plots/plot_results.py --runs "$OUT/runs/eval" --out "$OUT/plots/included_tool")
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
rows = ["| run | decis_mu | triad transitivity | both-null | one-sided | calls | LW post ref |", "|---|---|---|---|---|---|---|"]
for n in names:
    s, b = load(n); ref, how = lw_ref(n) if not n.startswith("lwpost_") else (None, "")
    bn, on, t = nulls(n) if not n.startswith("lwpost_") else ("–", "–", "–")
    refs = f"{ref:.4f} ({how})" if ref is not None else ("(is the reference)" if n.startswith("lwpost_") else "–")
    rows.append(f"| {n} | {b.get('decis_mu', float('nan')):.4f} | {b.get('transitivity_triad', float('nan')):.4f} | {bn} | {on} | {t} | {refs} |")
out.write_text("\n".join(rows) + "\n"); print("\n".join(rows))
PY
echo "plots: $(ls "$OUT"/plots/*.pdf "$OUT"/plots/included_tool/bars_decis_mu.png)"
