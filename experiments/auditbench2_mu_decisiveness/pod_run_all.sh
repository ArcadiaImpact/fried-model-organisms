#!/usr/bin/env bash
# One-command driver for the GPU measurement (SPEC.md "pod protocol"). Run ON the pod after pod_bootstrap.sh:
#   cd /workspace/fried-model-organisms/experiments/auditbench2_mu_decisiveness && HF_TOKEN=... ./pod_run_all.sh v1 v2
# Per set: filter adapters -> fetch (HF) -> [v2: build zero/random check adapters + key rewrite] -> serve (vLLM multi-LoRA)
#   -> concurrent-batch probe (PAR=PAR_OK if clean, else 1) -> [v2: zero-adapter key check] -> run_set.sh (prefill, items_2000)
#   -> sanity table -> kill server. Idempotent: run_set.sh skips models with a summary.json; fetch/check-adapter builds are no-ops on re-run.
# Env knobs: FILTER_V1 regex over adapter names (default 'sdfkto' -> 12 adapters = 4 quirks x {ab1orig, ab1post, ab2}),
#   FILTER_V2 (default 'kto_r64' -> the 4 KTO combos; use '.' for all 21), TP (default: GPU count), ITEMS (items_2000),
#   PAR_OK (6), SERVER_WAIT_MIN (45), EVAL_EXTRA (extra evalsuite args, e.g. '--concurrency 16'),
#   DRY=1 offline test against a mock endpoint (no fetch/serve/GPU): set ENDPOINT, EVALSUITE_CMD, RUNPY, ITEMS, OUT_ROOT, LOGS.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-/workspace/fried-model-organisms}; VPY=${VPY:-/workspace/vllm-venv/bin/python}
MODELS_ROOT=${MODELS_ROOT:-/workspace/hf/models}; OUT_ROOT=${OUT_ROOT:-/workspace/runs/eval}; LOGS=${LOGS:-/workspace/logs}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:${PORT:-8000}/v1}; ITEMS=${ITEMS:-items_2000}; PAR_OK=${PAR_OK:-6}
EVALSUITE_CMD=${EVALSUITE_CMD:-uv run evalsuite}; RUNPY=${RUNPY:-uv run python}   # both run from $REPO (its venv has openai)
FILTER_V1=${FILTER_V1:-sdfkto}; FILTER_V2=${FILTER_V2:-kto_r64}; DRY=${DRY:-0}; SERVER_WAIT_MIN=${SERVER_WAIT_MIN:-45}
export REPO MODELS_ROOT OUT_ROOT LOGS ENDPOINT EVALSUITE_CMD HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # the zero-adapter check loads the un-rewritten adapter at runtime (expected to be rejected)
mkdir -p "$OUT_ROOT" "$LOGS"
TS=$(date -u +%Y%m%dT%H%M%SZ); exec > >(tee -a "$LOGS/pod_run_all_$TS.log") 2>&1
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(v1 v2)
log(){ echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
[ "$DRY" = 1 ] || [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set (meta-llama/Llama-3.3-70B-Instruct is gated)"; exit 2; }
if [ -z "${TP:-}" ]; then if [ "$DRY" = 1 ]; then TP=1; else TP=$(nvidia-smi -L | wc -l); fi; fi
cd "$REPO"; log "commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) sets=${SETS[*]} TP=$TP ITEMS=$ITEMS PAR_OK=$PAR_OK DRY=$DRY OUT_ROOT=$OUT_ROOT"

filter_cfg(){ # $1 models json, $2 regex -> ${1%.json}.run.json with only the matching adapters
  python3 - "$1" "$2" <<'PY'
import json, re, sys
src, rx = sys.argv[1], sys.argv[2]; cfg = json.load(open(src))
keep = {k: v for k, v in cfg["adapters"].items() if re.search(rx, k)}
assert keep, f"filter {rx!r} matches no adapter in {src}"
out = src[:-5] + ".run.json"; json.dump({**cfg, "adapters": keep, "filter": rx}, open(out, "w"), indent=1)
print(f"{out}: {len(keep)}/{len(cfg['adapters'])} adapters: {' '.join(keep)}")
PY
}
prep_check_adapters(){ # $1 run.json (its .paths.json exists). Builds q36_noop/q36_rnd in agu18dec key format (SPEC am. 2),
  # rewrites copies with fix_lora_keys.py and serves the rewritten pair as extra LoRA modules (not evaluated by run_set.sh).
  local cfg=$1 paths=${1%.json}.paths.json base_dir ck=$MODELS_ROOT/check_adapters
  base_dir=$(python3 -c "import json;print(json.load(open('$paths'))['__base__'])"); mkdir -p "$ck"
  if [ ! -f "$ck/q36_rnd_fixed/.keys_rewritten.json" ]; then
    "$VPY" "$HERE/dryrun/make_tiny_lora.py" "$base_dir" "$ck" 80 q36_noop,q36_rnd
    for n in q36_noop q36_rnd; do
      rm -rf "$ck/${n}_fixed"; cp -r "$ck/$n" "$ck/${n}_fixed"
      "$VPY" "$HERE/fix_lora_keys.py" "$ck/${n}_fixed" base_model.model.model.layers. base_model.model.model.language_model.layers.
    done
  fi
  python3 - "$paths" "$ck" <<'PY'
import json, sys
p, ck = sys.argv[1], sys.argv[2]; d = json.load(open(p))
d["zz-noop-fixed"] = f"{ck}/q36_noop_fixed"; d["zz-rnd-fixed"] = f"{ck}/q36_rnd_fixed"
json.dump(d, open(p, "w"), indent=2); print("served LoRA modules:", len(d) - 1)
PY
}
serve_set(){ # $1 run.json, $2 tag -> prints the vLLM pid (setsid session leader; kill the whole group with kill -- -PID)
  local cfg=$1 tag=$2 extra=()
  if [ "$tag" = v2 ]; then export MAX_LORAS=${MAX_LORAS:-8}; extra=(--limit-mm-per-prompt '{"image":0,"video":0}')
  else export MAX_LORAS=${MAX_LORAS:-4}; fi   # 70B set: rank-128 adapters are ~1.6 GB each on the GPU
  "$HERE/serve_lora.sh" "$cfg" "$TP" 128 ${extra[@]+"${extra[@]}"} > "$LOGS/serve_$tag.txt"
  cat "$LOGS/serve_$tag.txt" >&2; grep -oE 'pid [0-9]+' "$LOGS/serve_$tag.txt" | grep -oE '[0-9]+'
}
wait_server(){ # $1 pid: wait for /v1/models (model load + CUDA graphs can take tens of minutes for the 70B) or die with the log tail
  local pid=$1 t=0
  until curl -sf "$ENDPOINT/models" >/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "vLLM (pid $pid) died; log tail:"; tail -n 40 "$LOGS"/vllm_*.log; return 1; }
    sleep 15; t=$((t + 15)); [ $t -gt $((SERVER_WAIT_MIN * 60)) ] && { echo "server not up after $SERVER_WAIT_MIN min"; return 1; }
  done
  log "server up after ${t}s: $(curl -sf "$ENDPOINT/models" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["data"]),"served names")')"
}
kill_server(){ # $1 pid: TERM the process group, wait for VRAM to drain, then hard-kill stragglers (engine cores hold VRAM)
  [ "$DRY" = 1 ] && return 0
  local pid=$1 used
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "${used:-0}" -lt 2000 ] && break; sleep 5
  done
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true; pkill -9 -f 'vllm serve' 2>/dev/null || true
  sleep 3; log "server down; max VRAM in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB"
}
sanity(){ # $1 run.json -> markdown table (decis_mu, LW-post reference where known, null-logprob rate) + file under OUT_ROOT
  python3 - "$1" "$OUT_ROOT" <<'PY' | tee "$OUT_ROOT/_table_$(basename "${1%.run.json}").md"
import json, sys, pathlib
cfg = json.load(open(sys.argv[1])); root = pathlib.Path(sys.argv[2])
ref = {"llama-3.3-70b-instruct": 0.8106, "ab1post-sdfkto-defer_to_users": 0.4366, "ab1post-sdfkto-flattery": 0.4116,
       "ab1post-sdfkto-reward_wireheading": 0.4847, "ab1post-sdfkto-secret_loyalty": 0.4647}  # LW post: bf16, items_2000, HF local logits
print("| model | decis_mu | LW post | both-null (one-sided) | calls |"); print("|---|---|---|---|---|")
for m in [cfg["served_base_name"], *cfg["adapters"]]:
    s = root / m / "summary.json"; c = root / m / "sentiment" / "calls.jsonl"
    if not s.exists(): print(f"| {m} | MISSING | {ref.get(m, '')} | | |"); continue
    sb = json.load(open(s)).get("benchmarks", {}).get("sentiment", {}); d = sb.get("decis_mu"); both = one = t = 0
    if d is None: print(f"| {m} | FAILED ({str(sb.get('error', 'no decis_mu'))[:60]}) | {ref.get(m, '')} | | {sum(1 for _ in open(c)) if c.exists() else 0} |"); continue
    for l in open(c):
        r = json.loads(l).get("raw", {}); t += 1; a = r.get("lpA") is None; b = r.get("lpB") is None; both += a and b; one += a != b
    flag = " SUSPECT" if (t == 0 or both / t > 0.01) else ""
    print(f"| {m} | {d:.4f} | {ref.get(m, '')} | {both}/{t}{flag} (one-sided {one}) | {t} |")
PY
}

for tag in "${SETS[@]}"; do
  case $tag in
    v1) src=$HERE/models_v1.json; rx=$FILTER_V1;;
    v2) src=$HERE/models_v2.json; rx=$FILTER_V2;;
    *) echo "unknown set '$tag' (v1|v2)"; exit 2;;
  esac
  log "[$tag] filter '$rx'"; filter_cfg "$src" "$rx"; cfg=${src%.json}.run.json
  BASE=$(python3 -c "import json;print(json.load(open('$cfg'))['served_base_name'])")
  ADS=$(python3 -c "import json;print(','.join(json.load(open('$cfg'))['adapters']))")
  if [ "$DRY" = 1 ]; then
    python3 -c "import json;c=json.load(open('$cfg'));json.dump({'__base__':'/dry/base',**{k:'/dry/'+k for k in c['adapters']}},open('${cfg%.json}.paths.json','w'),indent=1)"; PID=0
  else
    log "[$tag] fetch (base + $(tr ',' '\n' <<<"$ADS" | wc -l) adapters -> $MODELS_ROOT)"
    "$VPY" "$HERE/fetch_models.py" "$cfg" > "$LOGS/fetch_$tag.log" 2>&1 || { tail -n 30 "$LOGS/fetch_$tag.log"; exit 1; }; tail -n 3 "$LOGS/fetch_$tag.log"
    [ "$tag" = v2 ] && { log "[$tag] build zero/random check adapters (agu18dec key format) + rewrite"; prep_check_adapters "$cfg"; }
    log "[$tag] serve"; PID=$(serve_set "$cfg" "$tag"); wait_server "$PID" || exit 1
  fi
  log "[$tag] concurrent-batch probe (base prompts alongside adapter traffic; SPEC am. 4)"
  if $RUNPY "$HERE/dryrun/leak_test_concurrent.py" "$ENDPOINT" "$BASE" "$ADS" 60 32 > "$LOGS/probe_$tag.txt" 2>&1; then PAR=$PAR_OK; else PAR=1; fi
  tail -n 3 "$LOGS/probe_$tag.txt"; log "[$tag] probe -> PAR=$PAR"
  if [ "$tag" = v2 ] && [ "$DRY" != 1 ]; then
    log "[$tag] zero-adapter key-rewrite check (SPEC am. 2)"
    $RUNPY "$HERE/dryrun/zero_adapter_check.py" "$ENDPOINT" "$BASE" zz-noop-fixed zz-rnd-fixed "$MODELS_ROOT/check_adapters/q36_rnd" 2>&1 | tee "$LOGS/zero_adapter_check_$tag.txt" \
      || { echo "ZERO-ADAPTER CHECK FAILED: not running the $tag set (adapters would not be applied correctly)"; kill_server "$PID"; exit 1; }
  fi
  EXTRA=(--mode prefill --items-path "$ITEMS"); [ "$tag" = v2 ] && EXTRA+=(--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}')
  # shellcheck disable=SC2206
  [ -n "${EVAL_EXTRA:-}" ] && EXTRA+=($EVAL_EXTRA)
  log "[$tag] run_set PAR=$PAR ${EXTRA[*]}"; "$HERE/run_set.sh" "$cfg" "$PAR" "${EXTRA[@]}"
  log "[$tag] results"; sanity "$cfg"
  [ "$DRY" = 1 ] || kill_server "$PID"
done
log "ALL SETS DONE (${SETS[*]}) -> $OUT_ROOT (tables: $OUT_ROOT/_table_*.md; logs: $LOGS)"
