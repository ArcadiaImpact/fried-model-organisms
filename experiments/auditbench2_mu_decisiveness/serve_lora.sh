#!/usr/bin/env bash
# Serve base + all adapters of one model set with vLLM multi-LoRA on :8000 (loopback only).
# Usage: serve_lora.sh models_<set>.json [tp] [max_lora_rank] [extra vllm args...]
set -euo pipefail
CFG=$1; TP=${2:-2}; RANK=${3:-128}; shift 3 || shift $#
PATHS=${CFG%.json}.paths.json
BASE=$(python3 -c "import json;print(json.load(open('$PATHS'))['__base__'])")
NAME=$(python3 -c "import json;print(json.load(open('$CFG'))['served_base_name'])")
MODS=$(python3 -c "import json;p=json.load(open('$PATHS'));print(' '.join(f'{k}={v}' for k,v in p.items() if k!='__base__'))")
N=$(python3 -c "import json;print(len(json.load(open('$CFG'))['adapters']))")
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN} VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_NCCL_SYMM_MEM=0
mkdir -p /workspace/logs
setsid /workspace/vllm-venv/bin/vllm serve "$BASE" --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.92 \
  --enable-lora --max-lora-rank "$RANK" --max-loras 8 --max-cpu-loras "$((N+2))" \
  --lora-modules $MODS --port 8000 --host 127.0.0.1 "$@" > /workspace/logs/vllm_${NAME}.log 2>&1 < /dev/null &
echo "vllm pid $! ; log /workspace/logs/vllm_${NAME}.log"
