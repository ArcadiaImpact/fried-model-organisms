#!/usr/bin/env bash
# Sequential CPU dry run: fresh Qwen3-0.6B server -> LoRA leak probe -> evalsuite for base/noop/rnd (one model at a time).
cd /workspace/auditbench-2/cpu_test
KV_GB=2 ./start_vllm_cpu.sh Qwen3-0.6B qwen3-0.6b float32 noop=lora_noop rnd=lora_rnd > logs/q06_start.log 2>&1 || { echo "SERVER START FAILED"; tail -3 logs/q06_start.log; exit 1; }
echo "== leak probe (base/rnd/base/noop/base) =="; ./leak_test.sh 8001 qwen3-0.6b rnd qwen3-0.6b noop qwen3-0.6b
export EVALSUITE_CMD="uv run --no-project --python 3.12 --with numpy --with openai --with pyyaml --with python-dotenv --with huggingface_hub --with datasets --with-editable /workspace/fried-model-organisms evalsuite"
rm -rf /workspace/auditbench-2/cpu_test/runs2; mkdir -p /workspace/auditbench-2/cpu_test/runs2
ENDPOINT=http://127.0.0.1:8001/v1 OUT_ROOT=/workspace/auditbench-2/cpu_test/runs2 LOGS=/workspace/auditbench-2/cpu_test/logs \
  /workspace/fried-model-organisms/experiments/auditbench2_mu_decisiveness/run_set.sh /workspace/auditbench-2/cpu_test/models_cputest.json 1 \
  --mode prefill --items-path config/datasets/items.yaml --concurrency 16 --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' 2>&1 | tail -12
echo "== summaries =="; for f in runs2/*/summary.json; do echo "$f: $(jq -c '.benchmarks.sentiment.decis_mu' $f)"; done
