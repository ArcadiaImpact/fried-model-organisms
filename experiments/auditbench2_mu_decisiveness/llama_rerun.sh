#!/usr/bin/env bash
# llama_rerun.sh — pod-side recovery after the 10:53Z vLLM crash: drop failed runs (summary.json with "error"), re-serve the Llama set,
# re-run the missing models (run_set.sh skips finished ones), then the exact-logprob checks on the idle server, then kill the server.
# Env: TP (GPU count), EXACT=1 (run exact checks), EXACT_BASE_VARIANTS=nat,sp,fused, EXACT_ORG_NAMES="ab1post-sdfkto-defer_to_users …" (LoRA models to
# score exactly; each tested first on the 5k diag subset when DIAG_DIR exists), SERVER_WAIT_MIN (45). Log: $LOGS/llama_rerun_<ts>.log
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$EXP/../.." && pwd)
OUT_ROOT=${OUT_ROOT:-/workspace/runs/eval}; LOGS=${LOGS:-/workspace/logs}; ENDPOINT=${ENDPOINT:-http://127.0.0.1:8000/v1}
TP=${TP:-$(nvidia-smi -L | wc -l)}; SERVER_WAIT_MIN=${SERVER_WAIT_MIN:-45}; ITEMS=${ITEMS:-items_2000}
mkdir -p "$LOGS"; exec > >(tee -a "$LOGS/llama_rerun_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
export HF_HOME=${HF_HOME:-/workspace/hf}
for d in "$OUT_ROOT"/*/; do s="$d/summary.json"; if [ -f "$s" ] && grep -q '"error"' "$s"; then log "removing failed run $(basename "$d")"; rm -rf "$d"; fi; done
cd "$EXP"
if curl -sf "$ENDPOINT/models" >/dev/null && pgrep -f 'vllm serv[e]' >/dev/null; then
  PID=$(pgrep -f 'vllm serv[e]' | head -1); log "reusing the running Llama server (pid $PID): $(curl -sf "$ENDPOINT/models" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["data"]),"served names")')"
else
  pgrep -f 'vllm serv[e]' >/dev/null && { log "a vLLM server process exists but /v1/models is not answering — refusing to start another"; exit 1; }
  PID=$(MAX_LORAS=${MAX_LORAS:-4} ./serve_lora.sh "$EXP/models_v1.run.json" "$TP" 128 | sed -nE 's/^vllm pid ([0-9]+).*/\1/p'); log "serving Llama set, pid $PID"
  t=0; until curl -sf "$ENDPOINT/models" >/dev/null; do
    kill -0 "$PID" 2>/dev/null || { echo "vLLM died; log tail:"; tail -n 40 "$LOGS"/vllm_llama*.log; exit 1; }
    sleep 15; t=$((t + 15)); [ $t -gt $((SERVER_WAIT_MIN * 60)) ] && { echo "server not up after $SERVER_WAIT_MIN min"; exit 1; }
  done; log "server up after ${t}s: $(curl -sf "$ENDPOINT/models" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["data"]),"served names")')"
fi
# run_set.sh cd's into the repo, so the config path must be absolute (a relative path here silently ran nothing on 2026-10-03)
log "run_set (missing models only)"; ./run_set.sh "$EXP/models_v1.run.json" 1 --mode prefill --items-path "$ITEMS"; log "run_set exit $?"
for f in "$OUT_ROOT"/*/summary.json; do python3 -c "
import json,sys; s=json.load(open('$f')); b=s.get('benchmarks',{}).get('sentiment',{}); print(f\"{s['name']:40s} decis_mu {b.get('decis_mu')}  {b.get('error','')}\")"; done
if [ "${EXACT:-1}" = 1 ]; then
  log "exact check: parent (no LoRA)"; VARIANTS=${EXACT_BASE_VARIANTS:-nat,sp,fused} KEEP_SERVER=1 ./exact_check.sh llama-3.3-70b-instruct
  for n in ${EXACT_ORG_NAMES:-ab1post-sdfkto-defer_to_users ab1post-sdfkto-flattery}; do
    if [ -d "${DIAG_DIR:-/workspace/runs/diag/ab1post-defer-5k}" ] && [ "$n" = ab1post-sdfkto-defer_to_users ]; then
      log "exact check: LoRA crash test on the 5k diag subset ($n)"; rm -f "${DIAG_DIR:-/workspace/runs/diag/ab1post-defer-5k}"/sentiment/exact_ab_*.jsonl*
      (cd "$REPO" && uv run python "$EXP/exact_ab_logprobs.py" "$ENDPOINT" "$n" "${DIAG_DIR:-/workspace/runs/diag/ab1post-defer-5k}" 16) || { log "diag subset FAILED"; }
      curl -sf "$ENDPOINT/models" >/dev/null || { log "SERVER DIED during LoRA exact scoring — stopping exact checks (base result stands)"; break; }
    fi
    log "exact check: $n"; VARIANTS=${EXACT_ORG_VARIANTS:-nat,sp,fused} KEEP_SERVER=1 ./exact_check.sh "$n"
    curl -sf "$ENDPOINT/models" >/dev/null || { log "SERVER DIED during exact scoring of $n — stopping"; break; }
  done
fi
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true; sleep 20
pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true; pkill -9 -f 'vllm serv[e]' 2>/dev/null || true; sleep 3
log "server down; max VRAM in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB"
log "LLAMA RERUN DONE"
