#!/usr/bin/env bash
# upload_logs.sh [PULL_DIR] — ship this experiment's run logs off crab-factory: one tarball (pulled pod runs incl. calls.jsonl /
# edges.jsonl / exact_ab_logprobs.jsonl, pod logs, recovered-LW-post summaries, results/) plus a browsable copy of the small files
# (tables, plots, per-run summaries, exact comparisons, SPEC/RESULTS, code commit) to the private HF dataset
# arcadia-impact/sentiment-utility-logs under mo/auditbench2-mudecis/<ts>/ (ledger D19: GCS was the intended home, but no writable
# bucket is known from this VM; same layout convention as the LW post's mo/auditbench-llama70b/*_nogit.tar.gz).
# Verifies the remote listing against local sizes before declaring success. Env: WS, REPO_ID, PREFIX.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
WS=${WS:-/workspace/auditbench-2}; PULL=${1:-$(ls -d "$WS"/runs/pod_pull_*/ | sort | tail -1)}; PULL=${PULL%/}
REPO_ID=${REPO_ID:-arcadia-impact/sentiment-utility-logs}; PREFIX=${PREFIX:-mo/auditbench2-mudecis}
TS=$(date -u +%Y%m%dT%H%M%SZ); STAGE=$(mktemp -d "$WS/upload_stage_XXXX"); trap 'rm -rf "$STAGE"' EXIT
echo "pull: $PULL"; echo "dest: $REPO_ID :: $PREFIX/$TS"
REL=$(realpath --relative-to="$WS" "$PULL")
tar -C "$WS" -czf "$STAGE/auditbench2-mudecis_$TS.tar.gz" "$REL" results runs/eval $( [ -f "$WS/runs/diag_ab1post_defer_5k.jsonl" ] && echo runs/diag_ab1post_defer_5k.jsonl )
mkdir -p "$STAGE/browse"
for f in table.md exact_table.md plots exact runs; do [ -e "$WS/results/$f" ] && cp -r "$WS/results/$f" "$STAGE/browse/"; done
cp "$HERE"/SPEC.md "$HERE"/RESULTS.md "$STAGE/browse/"
{ echo "repo: ArcadiaImpact/fried-model-organisms"; echo "branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"; echo "commit: $(git -C "$REPO" rev-parse HEAD)"; echo "pull: $REL"; echo "uploaded: $TS"; } > "$STAGE/browse/PROVENANCE.txt"
du -sh "$STAGE"/* | sed 's/^/stage: /'
hf upload "$REPO_ID" "$STAGE" "$PREFIX/$TS" --repo-type dataset --commit-message "auditbench2 mu-decisiveness run logs ($TS)"
uv run --no-project --python 3.12 --with huggingface_hub python - "$REPO_ID" "$PREFIX/$TS" "$STAGE" <<'PY'
import os, pathlib, sys
from huggingface_hub import HfApi
repo, prefix, stage = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
remote = {f.path[len(prefix) + 1:]: getattr(f, "size", None) for f in HfApi().list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset", recursive=True) if hasattr(f, "size")}
local = {str(p.relative_to(stage)): p.stat().st_size for p in stage.rglob("*") if p.is_file()}
missing = sorted(set(local) - set(remote)); bad = sorted(k for k in local if k in remote and remote[k] != local[k])
print(f"remote files: {len(remote)}  local files: {len(local)}  missing: {len(missing)}  size mismatches: {len(bad)}")
for k in missing[:10]: print("  MISSING", k)
for k in bad[:10]: print("  SIZE", k, local[k], remote[k])
total = sum(remote.values()); print(f"remote total: {total/1e6:.1f} MB under {prefix}")
sys.exit(1 if missing or bad else 0)
PY
echo "UPLOAD VERIFIED: https://huggingface.co/datasets/$REPO_ID/tree/main/$PREFIX/$TS"
