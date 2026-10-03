#!/usr/bin/env bash
# qwen_merged_run.sh [adapter names…] — Qwen3.6-27B set with MERGED weights (SPEC D18). Default names: every adapter in
# models_v2.run.json (except the zz-* check adapters). Base is served plain first. Per adapter: merge (CPU, merge_qwen_adapter.py,
# verified) → serve plain bf16 with the same vLLM settings as the Llama set (no --enable-lora) → evalsuite (prefill, items_2000,
# enable_thinking=false) → kill → delete the merged copy (52 GB each; one at a time). MERGE_ONLY=1: just merge the given names
# (CPU-only, usable while the GPUs are busy). KEEP_MERGED=1 keeps merged weights. Log: $LOGS/qwen_merged_<ts>.log
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$EXP/../.." && pwd)
OUT_ROOT=${OUT_ROOT:-/workspace/runs/eval}; LOGS=${LOGS:-/workspace/logs}; ENDPOINT=${ENDPOINT:-http://127.0.0.1:8000/v1}
TP=${TP:-$(nvidia-smi -L | wc -l)}; ITEMS=${ITEMS:-items_2000}; SERVER_WAIT_MIN=${SERVER_WAIT_MIN:-45}; MERGED_ROOT=${MERGED_ROOT:-/workspace/hf/models/merged}
CFG=$EXP/models_v2.run.json; PATHS=${CFG%.json}.paths.json; VPY=/workspace/vllm-venv/bin/python
BASE=$(jq -r '.__base__' "$PATHS"); BASENAME=$(jq -r .served_base_name "$CFG")
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && mapfile -t NAMES < <(jq -r '.adapters|keys[]' "$CFG" | grep -v '^zz-')
mkdir -p "$LOGS" "$MERGED_ROOT"; exec > >(tee -a "$LOGS/qwen_merged_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
export PATH=/workspace/vllm-venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=${HF_HOME:-/workspace/hf} MU_TOP_LOGPROBS=${MU_TOP_LOGPROBS:-100}
merge(){ n=$1; AD=$(jq -r ".[\"$n\"]" "$PATHS"); M=$MERGED_ROOT/$n
  [ -f "$M/MERGE_OK" ] && { log "merged weights present for $n"; return 0; }
  rm -rf "$M"; log "merging $n ($AD)"; "$VPY" "$EXP/merge_qwen_adapter.py" "$BASE" "$AD" "$M" > "$LOGS/merge_$n.log" 2>&1
  if grep -q '^MERGE DONE$' "$LOGS/merge_$n.log" && grep -q -- '-> PASS' "$LOGS/merge_$n.log"; then touch "$M/MERGE_OK"; grep -E 'verify|saved' "$LOGS/merge_$n.log"; return 0; fi
  log "MERGE FAILED $n"; tail -n 8 "$LOGS/merge_$n.log"; rm -rf "$M"; return 1; }
if [ "${MERGE_ONLY:-0}" = 1 ]; then for n in "${NAMES[@]}"; do merge "$n"; done; log "MERGE ONLY DONE"; exit 0; fi
serve(){ setsid vllm serve "$1" --served-model-name "$2" --tensor-parallel-size "$TP" --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.92 --max-logprobs 128 --limit-mm-per-prompt '{"image":0,"video":0}' --port 8000 --host 127.0.0.1 \
  > "$LOGS/vllm_$2.log" 2>&1 < /dev/null & echo $!; }
wait_server(){ t=0; until curl -sf "$ENDPOINT/models" >/dev/null; do
    kill -0 "$1" 2>/dev/null || { echo "vLLM died; log tail:"; tail -n 30 "$LOGS/vllm_$2.log"; return 1; }
    sleep 15; t=$((t + 15)); [ $t -gt $((SERVER_WAIT_MIN * 60)) ] && { echo "server not up after $SERVER_WAIT_MIN min"; return 1; }
  done; log "server up after ${t}s ($2)"; }
kill_server(){ kill -TERM -- "-$1" 2>/dev/null || kill -TERM "$1" 2>/dev/null || true
  for _ in $(seq 1 24); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "${used:-0}" -lt 2000 ] && break; sleep 5; done
  pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true; pkill -9 -f 'vllm serv[e]' 2>/dev/null || true; sleep 3; log "server down ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB)"; }
eval_one(){ m=$1; [ -f "$OUT_ROOT/$m/summary.json" ] && { log "skip $m (done)"; return; }
  (cd "$REPO" && OPENAI_API_KEY=EMPTY uv run evalsuite --endpoint "$ENDPOINT" --model "$m" --name "$m" --benchmarks sentiment --items-path "$ITEMS" \
     --concurrency 96 --out-root "$OUT_ROOT" --mode prefill --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' > "$LOGS/eval_$m.log" 2>&1) || { log "FAILED $m (see $LOGS/eval_$m.log)"; return; }
  python3 - "$OUT_ROOT/$m/sentiment/calls.jsonl" <<'PY' || echo "SUSPECT $m (high both-null rate)"
import json, sys
both = one = t = 0
for l in open(sys.argv[1]):
    r = json.loads(l).get("raw", {}); t += 1; a = r.get("lpA") is None; b = r.get("lpB") is None; both += a and b; one += a != b
print(f"both-null (p_a=0.5 fallback): {both}/{t} ({100*both/max(t,1):.2f}%); one-sided (letter below top-N): {one}/{t} ({100*one/max(t,1):.2f}%)")
sys.exit(1 if t == 0 or both / t > 0.01 else 0)
PY
  echo "done $m $(date -u +%T)"; }
pgrep -f 'vllm serv[e]' >/dev/null && { log "a vLLM server is already running — refusing to start"; exit 1; }
if [ ! -f "$OUT_ROOT/$BASENAME/summary.json" ]; then
  PID=$(serve "$BASE" "$BASENAME"); wait_server "$PID" "$BASENAME" && eval_one "$BASENAME"; kill_server "$PID"
fi
for n in "${NAMES[@]}"; do
  [ -f "$OUT_ROOT/$n/summary.json" ] && { log "skip $n (done)"; continue; }
  merge "$n" || continue; M=$MERGED_ROOT/$n
  PID=$(serve "$M" "$n"); wait_server "$PID" "$n" && eval_one "$n"; kill_server "$PID"
  [ "${KEEP_MERGED:-0}" = 1 ] || rm -rf "$M"
done
log "QWEN MERGED DONE"
