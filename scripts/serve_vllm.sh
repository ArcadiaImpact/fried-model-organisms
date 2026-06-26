#!/usr/bin/env bash
# Serve a HuggingFace model with vLLM's OpenAI-compatible server, so `mu-evalsuite --endpoint
# http://localhost:<port>/v1` (and `mu-decisiveness --base-url ...`) can drive it. vLLM exposes
# the prompt/echo logprobs the FULL suite needs (loglikelihood MMLU + perplexity).
#
# vLLM pins a different (cu128) torch + transformers<5 than the metric package, so it lives in
# its OWN venv (.venv-vllm), created on first run. This script does NOT touch your project venv.
#
# Usage:
#   bash scripts/serve_vllm.sh <hf-model-id-or-path> [--port 8000] [--tp 1] \
#        [--max-model-len 8192] [--gpu-mem-util 0.90] [--venv .venv-vllm] [--skip-setup]
#
# Requires a CUDA 12.8-capable host (RunPod A100/H100 pool, etc.) and `uv`. For gated models
# run `huggingface-cli login` first.
set -euo pipefail

MODEL="${1:?usage: serve_vllm.sh <hf-model> [--port N] [--tp N] [--max-model-len N] [--gpu-mem-util F] [--venv DIR] [--skip-setup]}"
shift || true

PORT=8000; TP=1; MAX_MODEL_LEN=""; GPU_MEM_UTIL=0.90; VENV=".venv-vllm"; SKIP_SETUP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --skip-setup) SKIP_SETUP=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [ "$SKIP_SETUP" != 1 ] && [ ! -x "$VENV/bin/vllm" ]; then
  echo "[serve_vllm] creating $VENV (vllm==0.11.0, cu128, transformers<5; FlashInfer removed)"
  uv venv "$VENV" --python 3.11
  uv pip install --python "$VENV/bin/python" --torch-backend=cu128 "vllm==0.11.0"
  uv pip install --python "$VENV/bin/python" "transformers<5"   # vllm 0.11 needs the 4.x tokenizer
  # FlashInfer can't JIT without nvcc/ninja on many hosts -> remove it and use native fallbacks.
  uv pip uninstall --python "$VENV/bin/python" flashinfer-python flashinfer-cubin 2>/dev/null || true
  rm -rf "$VENV"/lib/python*/site-packages/flashinfer || true
fi

# Native (no-JIT) attention/sampler + disable the symmetric-memory all-reduce that breaks TP>1
# in many containers. All overridable from the environment.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM:-0}"

args=(serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP"
      --gpu-memory-utilization "$GPU_MEM_UTIL" --enforce-eager)
[ -n "$MAX_MODEL_LEN" ] && args+=(--max-model-len "$MAX_MODEL_LEN")

echo "[serve_vllm] $VENV/bin/vllm ${args[*]}"
exec "$VENV/bin/vllm" "${args[@]}"
