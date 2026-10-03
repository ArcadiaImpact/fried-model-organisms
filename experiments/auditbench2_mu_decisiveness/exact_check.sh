#!/usr/bin/env bash
# exact_check.sh [served names...] — pod-side robustness check (SPEC D15): re-serve the Llama set and re-score every Elo edge of the
# listed finished runs with EXACT A/B logprobs (vLLM prompt_logprobs), then re-fit Case-V → sentiment/exact_ab_summary.json per run.
# Quantifies the top-N logprob truncation of the main run (p_a saturating at 0/1 when the losing letter falls below top-100).
# Env: VARIANTS=nat,sp[,fused] (default: all three), CONC (32), CFG, TP, OUT_ROOT, LOGS, KEEP_SERVER=1.
# Default names: parent + one organism per arm (flattery). ~25 min per 70B model at concurrency 64. Logs: $LOGS/exact_check_<ts>.log
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$EXP/../.." && pwd)
OUT_ROOT=${OUT_ROOT:-/workspace/runs/eval}; LOGS=${LOGS:-/workspace/logs}; ENDPOINT=${ENDPOINT:-http://127.0.0.1:8000/v1}
CFG=${CFG:-$EXP/models_v1.run.json}; TP=${TP:-$(nvidia-smi -L | wc -l)}; CONC=${CONC:-32}; SERVER_WAIT_MIN=${SERVER_WAIT_MIN:-45}
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=(llama-3.3-70b-instruct ab1post-sdfkto-flattery ab2-sdfkto-flattery ab1orig-sdfkto-flattery)
mkdir -p "$LOGS"; exec > >(tee -a "$LOGS/exact_check_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
export HF_HOME=${HF_HOME:-/workspace/hf}
if curl -sf "$ENDPOINT/models" >/dev/null; then log "server already up"; PID=""; else
  cd "$EXP"; PID=$(MAX_LORAS=${MAX_LORAS:-4} ./serve_lora.sh "$CFG" "$TP" 128 | sed -nE 's/^vllm pid ([0-9]+).*/\1/p'); log "serving $CFG pid $PID"
  t=0; until curl -sf "$ENDPOINT/models" >/dev/null; do
    kill -0 "$PID" 2>/dev/null || { echo "vLLM died; log tail:"; tail -n 40 "$LOGS"/vllm_*.log; exit 1; }
    sleep 15; t=$((t + 15)); [ $t -gt $((SERVER_WAIT_MIN * 60)) ] && { echo "server not up after $SERVER_WAIT_MIN min"; exit 1; }
  done; log "server up after ${t}s"
fi
cd "$REPO"; rc=0
for n in "${NAMES[@]}"; do
  d="$OUT_ROOT/$n"; [ -f "$d/sentiment/edges.jsonl" ] || { log "SKIP $n (no edges.jsonl)"; continue; }
  log "exact scoring $n"; uv run python "$EXP/exact_ab_logprobs.py" "$ENDPOINT" "$n" "$d" "$CONC" ${VARIANTS:+--variants=$VARIANTS} || { log "FAILED $n"; rc=1; }
  [ -f "$d/sentiment/exact_ab_summary.json" ] && { echo "--- $n"; cat "$d/sentiment/exact_ab_summary.json"; echo; }
done
log "summary:"; for n in "${NAMES[@]}"; do f="$OUT_ROOT/$n/sentiment/exact_ab_summary.json"; [ -f "$f" ] || continue; python3 - "$f" "$n" <<'PYS'
import json, sys
s = json.load(open(sys.argv[1])); g = lambda k: s[k] if s.get(k) is not None else float("nan")
print(f"{sys.argv[2]:40s} run {g('decis_run_refit'):.4f}  sum {g('decis_sum'):.4f}  max {g('decis_max'):.4f}  fused(post method) {g('decis_fused'):.4f}  saturated_run {s.get('n_saturated_run')}")
PYS
done
if [ -n "${PID:-}" ] && [ "${KEEP_SERVER:-0}" != 1 ]; then
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true; sleep 20
  pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true; pkill -9 -f 'vllm serv[e]' 2>/dev/null || true; sleep 3
  log "server down; max VRAM in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB"
fi
log "EXACT CHECK DONE rc=$rc"; exit $rc
