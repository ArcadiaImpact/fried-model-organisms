#!/usr/bin/env bash
# llama_fix_nulls.sh — run AFTER the Qwen pass (GPUs free). Exact letter logprobs for the Llama set WITHOUT vLLM's prompt_logprobs path
# (it crashes the engine for LoRA models under load, 2026-10-03 ×2). Uses exact_ab_logprobs.py --forced (forced-letter chat calls,
# raw logprobs; verified equal to prompt_logprobs on a CPU vLLM for base and LoRA):
#  (1) --only-null --variants=max for every run with a non-negligible null rate: the null edges (both-null → p_a=0.5 fallback, one-sided →
#      p_a saturated at 0/1) get their exact p_max; the other edges' run p_a already equals exact p_max → decis_hybrid_max = exact p_max.
#  (2) --variants=fused --max-edges=5000 (seeded subset, all edge types) for the ab1post reward_wireheading / secret_loyalty organisms,
#      where this run's top-100 number is 0.2 above the LW post's: reproduces the post's fused-token convention on a subset.
# Kills the server at the end. Env: NULL_NAMES, FUSED_NAMES, CONC (128), FUSED_N (5000), CFG, LOGS.
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); LOGS=${LOGS:-/workspace/logs}; ENDPOINT=${ENDPOINT:-http://127.0.0.1:8000/v1}
CONC=${CONC:-128}; FUSED_N=${FUSED_N:-5000}
NULL_NAMES=${NULL_NAMES:-"llama-3.3-70b-instruct ab1post-sdfkto-secret_loyalty ab2-sdfkto-secret_loyalty ab1post-sdfkto-reward_wireheading ab2-sdfkto-reward_wireheading ab1orig-sdfkto-secret_loyalty ab1post-sdfkto-flattery ab1post-sdfkto-defer_to_users ab2-sdfkto-flattery"}
FUSED_NAMES=${FUSED_NAMES:-"ab1post-sdfkto-reward_wireheading ab1post-sdfkto-secret_loyalty"}
mkdir -p "$LOGS"; exec > >(tee -a "$LOGS/llama_fix_nulls_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
log "only-null forced max: $NULL_NAMES | fused subset ($FUSED_N edges): $FUSED_NAMES | CONC $CONC"
for n in $NULL_NAMES; do
  VARIANTS=max EXTRA_ARGS="--forced --only-null" KEEP_SERVER=1 CONC=$CONC "$EXP/exact_check.sh" "$n" || log "FAILED only-null $n"
  curl -sf "$ENDPOINT/models" >/dev/null || { log "SERVER DIED during $n; stopping"; exit 1; }
done
for n in $FUSED_NAMES; do
  VARIANTS=fused EXTRA_ARGS="--forced --max-edges=$FUSED_N" KEEP_SERVER=1 CONC=$CONC "$EXP/exact_check.sh" "$n" || log "FAILED fused $n"
  curl -sf "$ENDPOINT/models" >/dev/null || { log "SERVER DIED during fused $n; stopping"; exit 1; }
done
log "stopping server"; pkill -TERM -f 'vllm serv[e]' 2>/dev/null || true; sleep 20
pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true; pkill -9 -f 'vllm serv[e]' 2>/dev/null || true; sleep 3
log "server down; VRAM in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB"
log "LLAMA FIX NULLS DONE"
