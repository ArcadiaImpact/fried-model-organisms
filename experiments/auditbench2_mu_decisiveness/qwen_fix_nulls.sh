#!/usr/bin/env bash
# qwen_fix_nulls.sh [adapter names…] — forced-letter exact rescoring (exact_ab_logprobs.py --forced --only-null, SPEC D20) of the
# null Elo edges of the Qwen3.6-27B runs. The base is served plain; each adapter is re-merged (CPU, ~100 s, 55 GB — one at a time,
# the disk holds one copy) and served plain with the SAME vLLM settings as qwen_merged_run.sh; the chat requests carry the eval's
# extra body (enable_thinking=false) so the scored prefix is identical to the run's. Default names: the two adapters whose
# null-sensitivity bound (results/null_sensitivity.jsonl) was widest. Refuses to start while another vLLM server is running.
# Env: CONC (128), VARIANTS (max), EXTRA_BODY, KEEP_MERGED=1. Log: $LOGS/qwen_fix_nulls_<ts>.log; end marker 'QWEN FIX NULLS DONE'.
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$EXP/../.." && pwd)
OUT_ROOT=${OUT_ROOT:-/workspace/runs/eval}; LOGS=${LOGS:-/workspace/logs}; ENDPOINT=${ENDPOINT:-http://127.0.0.1:8000/v1}
TP=${TP:-$(nvidia-smi -L | wc -l)}; SERVER_WAIT_MIN=${SERVER_WAIT_MIN:-45}; MERGED_ROOT=${MERGED_ROOT:-/workspace/hf/models/merged}
CFG=$EXP/models_v2.run.json; PATHS=${CFG%.json}.paths.json; VPY=/workspace/vllm-venv/bin/python
BASE=$(jq -r '.__base__' "$PATHS"); BASENAME=$(jq -r .served_base_name "$CFG")
CONC=${CONC:-128}; VARIANTS=${VARIANTS:-max}; EXTRA_BODY=${EXTRA_BODY:-'{"chat_template_kwargs":{"enable_thinking":false}}'}
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=(ab2-flattery_tdkto_r64 ab2-hardcode_test_cases_tdkto_r64)
mkdir -p "$LOGS" "$MERGED_ROOT"; exec > >(tee -a "$LOGS/qwen_fix_nulls_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
export PATH=/workspace/vllm-venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=${HF_HOME:-/workspace/hf}
# merge / serve / wait_server / kill_server are the recipes of qwen_merged_run.sh, verbatim, so the exact pass sees the run's model.
merge(){ n=$1; AD=$(jq -r ".[\"$n\"]" "$PATHS"); M=$MERGED_ROOT/$n
  [ -f "$M/MERGE_OK" ] && { log "merged weights present for $n"; return 0; }
  rm -rf "$M"; log "merging $n ($AD)"; "$VPY" "$EXP/merge_qwen_adapter.py" "$BASE" "$AD" "$M" > "$LOGS/merge_$n.log" 2>&1
  if grep -q '^MERGE DONE$' "$LOGS/merge_$n.log" && grep -q -- '-> PASS' "$LOGS/merge_$n.log"; then touch "$M/MERGE_OK"; grep -E 'verify|saved' "$LOGS/merge_$n.log"; return 0; fi
  log "MERGE FAILED $n"; tail -n 8 "$LOGS/merge_$n.log"; rm -rf "$M"; return 1; }
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
score(){ n=$1; d=$OUT_ROOT/$n; [ -f "$d/sentiment/edges.jsonl" ] || { log "SKIP $n (no edges.jsonl)"; return 1; }
  log "exact scoring $n (forced, only-null, variants $VARIANTS, extra body $EXTRA_BODY)"
  (cd "$REPO" && uv run python "$EXP/exact_ab_logprobs.py" "$ENDPOINT" "$n" "$d" "$CONC" --forced --only-null "--variants=$VARIANTS" "--extra-body=$EXTRA_BODY") || { log "FAILED $n"; return 1; }
  python3 - "$d/sentiment/exact_ab_summary.json" "$n" <<'PYS'
import json, sys
s = json.load(open(sys.argv[1])); g = lambda k: s[k] if s.get(k) is not None else float("nan")
print(f"{sys.argv[2]:40s} run {g('decis_run_refit'):.4f}  hybrid_max {g('decis_hybrid_max'):.4f}  hybrid_fused {g('decis_hybrid_fused'):.4f}  null both/one {s.get('n_null_both')}/{s.get('n_null_one')}  scored {s.get('n_scored')}  forced misses {s.get('n_forced_miss')}")
PYS
}
done_already(){ f=$OUT_ROOT/$1/sentiment/exact_ab_summary.json; [ -f "$f" ] && jq -e '.method == "forced" and .only_null == true and .decis_hybrid_max != null' "$f" >/dev/null 2>&1; }
pgrep -f 'vllm serv[e]' >/dev/null && { log "a vLLM server is already running — refusing to start"; exit 1; }
if done_already "$BASENAME"; then log "skip $BASENAME (forced only-null summary present)"; else
  PID=$(serve "$BASE" "$BASENAME"); wait_server "$PID" "$BASENAME" && score "$BASENAME"; kill_server "$PID"
fi
for n in "${NAMES[@]}"; do
  done_already "$n" && { log "skip $n (forced only-null summary present)"; continue; }
  merge "$n" || continue; M=$MERGED_ROOT/$n
  PID=$(serve "$M" "$n"); wait_server "$PID" "$n" && score "$n"; kill_server "$PID"
  [ "${KEEP_MERGED:-0}" = 1 ] || rm -rf "$M"
done
log "QWEN FIX NULLS DONE"
