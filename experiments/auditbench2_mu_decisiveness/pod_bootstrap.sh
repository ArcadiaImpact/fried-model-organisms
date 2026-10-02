#!/usr/bin/env bash
# One-time pod bootstrap: uv, a vLLM venv, the fried-model-organisms repo (this branch) with the
# `api` extra, and the HF cache dir. Run ON the pod as root. Idempotent.
#   HF_TOKEN must be in the environment (export it from the controlling shell via ssh env-stdin).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
WORK=/workspace; mkdir -p $WORK/hf $WORK/runs $WORK/logs
export HF_HOME=$WORK/hf
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH)
export PATH=$HOME/.local/bin:$PATH
# vLLM venv (own torch). vllm==0.29.0 is REQUIRED: Qwen/Qwen3.6-27B is `Qwen3_5ForConditionalGeneration`
# (unknown to the 0.11.0 recipe in scripts/serve_vllm.sh). 0.29.0 wheels: PyPI default = CUDA 13.0 (torch 2.13.0
# cu130, driver >= 580); GitHub release ships a +cu129 variant for 12.9 drivers (torch cu129). No cu128 build exists,
# so the pod host must report CUDA >= 12.9 (`create-pod.sh` MIN_CUDA_VERSION=12.9).
VLLM_VER=${VLLM_VER:-0.29.0}
if [ ! -x $WORK/vllm-venv/bin/vllm ]; then
  DRV_CUDA=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "driver CUDA: ${DRV_CUDA:-unknown}"
  uv venv --python 3.12 $WORK/vllm-venv
  if [ -n "$DRV_CUDA" ] && [ "$(printf '%s\n' 13.0 "$DRV_CUDA" | sort -V | head -1)" = "13.0" ]; then
    uv pip install --python $WORK/vllm-venv/bin/python --torch-backend=cu130 "vllm==$VLLM_VER" "huggingface_hub>=0.34" hf_transfer
  else
    WHL="https://github.com/vllm-project/vllm/releases/download/v$VLLM_VER/vllm-$VLLM_VER+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    uv pip install --python $WORK/vllm-venv/bin/python --torch-backend=cu129 "vllm @ $WHL" "huggingface_hub>=0.34" hf_transfer
  fi
fi
# metric repo (expects a tarball of the branch at /workspace/fmo.tar, shipped via tar-over-ssh)
if [ ! -d $WORK/fried-model-organisms ]; then mkdir -p $WORK/fried-model-organisms && tar -xf $WORK/fmo.tar -C $WORK/fried-model-organisms; fi
cd $WORK/fried-model-organisms && uv sync --extra api --extra dev -q
$WORK/vllm-venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK', torch.cuda.device_count(), 'GPUs; torch', torch.__version__)"
$WORK/vllm-venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
echo "BOOTSTRAP DONE"
