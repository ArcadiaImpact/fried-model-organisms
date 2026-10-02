#!/usr/bin/env bash
# Run the mu-decisiveness `sentiment` benchmark (evalsuite) for the base + every adapter in a
# model set, PAR at a time, against the local vLLM. Writes runs/eval/<name>/summary.json.
# Usage: run_set.sh models_<set>.json [PAR] [extra evalsuite args...]   e.g. --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
set -euo pipefail
CFG=$1; PAR=${2:-4}; shift 2 || shift $#
# Overridable for local dry-runs: ENDPOINT (OpenAI-compatible base URL), OUT_ROOT, EVALSUITE_CMD, REPO.
REPO=${REPO:-/workspace/fried-model-organisms}; ENDPOINT=${ENDPOINT:-http://127.0.0.1:8000/v1}
OUT_ROOT=${OUT_ROOT:-/workspace/runs/eval}; LOGS=${LOGS:-/workspace/logs}; EVALSUITE_CMD=${EVALSUITE_CMD:-uv run evalsuite}
export REPO ENDPOINT OUT_ROOT LOGS EVALSUITE_CMD
cd "$REPO"
NAME=$(python3 -c "import json;print(json.load(open('$CFG'))['served_base_name'])")
MODELS=("$NAME" $(python3 -c "import json;print(' '.join(json.load(open('$CFG'))['adapters']))"))
mkdir -p "$OUT_ROOT" "$LOGS"
until curl -sf "$ENDPOINT/models" >/dev/null; do sleep 10; done
run_one() { m=$1; shift
  [ -f "$OUT_ROOT/$m/summary.json" ] && { echo "skip $m (done)"; return; }
  OPENAI_API_KEY=EMPTY $EVALSUITE_CMD --endpoint "$ENDPOINT" --model "$m" --name "$m" \
     --benchmarks sentiment --items-path items_500 --concurrency 96 --out-root "$OUT_ROOT" "$@" \
     > "$LOGS/eval_$m.log" 2>&1 && echo "done $m $(date -u +%H:%M:%S)" || echo "FAILED $m (see $LOGS/eval_$m.log)"; }
export -f run_one
printf '%s\n' "${MODELS[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {} "$@"
echo "SET DONE $CFG"
