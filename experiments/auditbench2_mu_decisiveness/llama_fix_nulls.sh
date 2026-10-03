#!/usr/bin/env bash
# llama_fix_nulls.sh — run AFTER the Qwen pass (GPUs free). Re-serves the Llama set (exact_check.sh starts it) and
#  (1) fully re-scores every name in $FULL (default: the runs with massive top-100 truncation — ab1post reward_wireheading 65 % one-sided,
#      ab2 reward_wireheading 8.5 % both-null + 45 % one-sided, ab1post secret_loyalty 12 % both-null + 69 % one-sided; all flagged SUSPECT)
#      with the three letter conventions (nat, sp, fused = the LW post's), and
#  (2) re-scores ONLY the null edges (both-null → p_a = 0.5 fallback; one-sided → p_a saturated at 0/1) of every other organism in the
#      set with nat,sp and re-fits the hybrid (run p_util with those edges replaced by the exact p) → decis_hybrid_* in each run's
#      sentiment/exact_ab_summary.json. Kills the server at the end. Env: FULL, CONC (48), CFG, LOGS, SKIP (names to skip).
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); LOGS=${LOGS:-/workspace/logs}; CFG=${CFG:-$EXP/models_v1.run.json}; ENDPOINT=${ENDPOINT:-http://127.0.0.1:8000/v1}
FULL=${FULL:-"ab1post-sdfkto-reward_wireheading ab2-sdfkto-reward_wireheading ab1post-sdfkto-secret_loyalty"}; CONC=${CONC:-64}
mkdir -p "$LOGS"; exec > >(tee -a "$LOGS/llama_fix_nulls_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
mapfile -t ALL < <(jq -r '.adapters | keys[]' "$CFG" | grep -v '^zz-')
OTHERS=(); for n in "${ALL[@]}"; do case " $FULL " in *" $n "*) continue;; esac; case " ${SKIP:-} " in *" $n "*) continue;; esac; OTHERS+=("$n"); done
log "full exact (nat,sp,fused): $FULL | only-null (nat,sp): ${OTHERS[*]}"
for n in $FULL; do
  VARIANTS=nat,sp,fused KEEP_SERVER=1 CONC=$CONC "$EXP/exact_check.sh" "$n" || log "FAILED full $n"
  curl -sf "$ENDPOINT/models" >/dev/null || { log "SERVER DIED after the full exact pass of $n; stopping"; exit 1; }
done
VARIANTS=nat,sp EXTRA_ARGS=--only-null KEEP_SERVER=1 CONC=$CONC "$EXP/exact_check.sh" "${OTHERS[@]}" || log "FAILED only-null batch"
log "stopping server"; pkill -TERM -f 'vllm serv[e]' 2>/dev/null || true; sleep 20
pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true; pkill -9 -f 'vllm serv[e]' 2>/dev/null || true; sleep 3
log "server down; VRAM in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB"
log "LLAMA FIX NULLS DONE"
