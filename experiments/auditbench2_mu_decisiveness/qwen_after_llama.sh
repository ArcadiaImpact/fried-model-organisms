#!/usr/bin/env bash
# qwen_after_llama.sh [names…] — chain step: wait for the Llama fix pass (llama_fix_nulls.sh) to exit, stop any vLLM server it left
# behind (it normally kills its own), then run qwen_fix_nulls.sh. Launch detached from $EXP: (nohup setsid bash ./qwen_after_llama.sh &)
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); LOGS=${LOGS:-/workspace/logs}
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
log "waiting for the Llama fix pass to exit"
while [ "$(pgrep -fc 'bash ./llama_fix_null[s]')" != 0 ]; do sleep 30; done
log "Llama driver gone; last Llama log line: $(tail -n 1 "$(ls -t "$LOGS"/llama_fix_nulls_*.log | head -1)")"
if pgrep -f 'vllm serv[e]' >/dev/null; then
  log "vLLM server still up — stopping it"; pkill -TERM -f 'vllm serv[e]' || true; sleep 30
  pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true; pkill -9 -f 'vllm serv[e]' 2>/dev/null || true; sleep 5
fi
log "GPU memory in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB; starting qwen_fix_nulls.sh"
cd "$EXP" && bash ./qwen_fix_nulls.sh "$@"; log "qwen_after_llama exit $?"
