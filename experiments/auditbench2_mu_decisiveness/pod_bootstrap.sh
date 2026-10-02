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
# vLLM venv (own torch); pin 0.11.0 as in scripts/serve_vllm.sh of the metric repo.
if [ ! -x $WORK/vllm-venv/bin/vllm ]; then
  uv venv --python 3.12 $WORK/vllm-venv
  uv pip install --python $WORK/vllm-venv/bin/python --torch-backend=cu128 "vllm==0.11.0" "huggingface_hub>=0.34" hf_transfer
  uv pip install --python $WORK/vllm-venv/bin/python "transformers<5"
  uv pip uninstall --python $WORK/vllm-venv/bin/python flashinfer-python flashinfer-cubin 2>/dev/null || true
  rm -rf $WORK/vllm-venv/lib/python*/site-packages/flashinfer || true
fi
# metric repo (expects a tarball of the branch at /workspace/fmo.tar, shipped via tar-over-ssh)
if [ ! -d $WORK/fried-model-organisms ]; then mkdir -p $WORK/fried-model-organisms && tar -xf $WORK/fmo.tar -C $WORK/fried-model-organisms; fi
cd $WORK/fried-model-organisms && uv sync --extra api --extra dev -q
$WORK/vllm-venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK', torch.cuda.device_count(), 'GPUs')"
echo "BOOTSTRAP DONE"
