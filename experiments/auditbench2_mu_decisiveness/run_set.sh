#!/usr/bin/env bash
# Run the mu-decisiveness `sentiment` benchmark (evalsuite) for the base + every adapter in a
# model set, PAR at a time, against the local vLLM. Writes runs/eval/<name>/summary.json.
# Usage: run_set.sh models_<set>.json [PAR] [extra evalsuite args...]   e.g. --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
set -euo pipefail
CFG=$1; PAR=${2:-4}; shift 2 || shift $#
cd /workspace/fried-model-organisms
NAME=$(python3 -c "import json;print(json.load(open('$CFG'))['served_base_name'])")
MODELS=("$NAME" $(python3 -c "import json;print(' '.join(json.load(open('$CFG'))['adapters']))"))
mkdir -p /workspace/runs/eval /workspace/logs
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 10; done
run_one() { m=$1; shift
  [ -f /workspace/runs/eval/$m/summary.json ] && { echo "skip $m (done)"; return; }
  OPENAI_API_KEY=EMPTY uv run evalsuite --endpoint http://127.0.0.1:8000/v1 --model "$m" --name "$m" \
     --benchmarks sentiment --items-path items_500 --concurrency 96 --out-root /workspace/runs/eval "$@" \
     > /workspace/logs/eval_$m.log 2>&1 && echo "done $m $(date -u +%H:%M:%S)" || echo "FAILED $m"; }
export -f run_one
printf '%s\n' "${MODELS[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {} "$@"
echo "SET DONE $CFG"
