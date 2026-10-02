#!/usr/bin/env bash
# Fresh server -> concurrent leak probe alone -> noop-only eval (one model at a time) -> summary.
cd /workspace/auditbench-2/cpu_test
docker rm -f vllm-cpu >/dev/null 2>&1
KV_GB=2 ./start_vllm_cpu.sh Qwen3-0.6B qwen3-0.6b float32 noop=lora_noop rnd=lora_rnd > logs/q06_start2.log 2>&1 || { echo "SERVER START FAILED"; tail -3 logs/q06_start2.log; exit 1; }
echo "== concurrent leak probe (fresh idle server) =="
uv run --no-project --python 3.12 --with openai python leak_test_concurrent.py http://127.0.0.1:8001/v1 qwen3-0.6b noop,rnd 60 32 2>&1 | tail -6
echo "== noop-only eval =="
export EVALSUITE_CMD="uv run --no-project --python 3.12 --with numpy --with openai --with pyyaml --with python-dotenv --with huggingface_hub --with datasets --with-editable /workspace/fried-model-organisms evalsuite"
mkdir -p runs5
ENDPOINT=http://127.0.0.1:8001/v1 OUT_ROOT=/workspace/auditbench-2/cpu_test/runs5 LOGS=/workspace/auditbench-2/cpu_test/logs5 \
  /workspace/fried-model-organisms/experiments/auditbench2_mu_decisiveness/run_set.sh /workspace/auditbench-2/cpu_test/models_cputest_nooponly.json 1 \
  --mode prefill --items-path config/datasets/items.yaml --concurrency 16 --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' 2>&1 | grep -E "null lpA|done |FAILED|SUSPECT|SET DONE"
echo "noop decis: $(jq -c '.benchmarks.sentiment.decis_mu' runs5/noop/summary.json 2>/dev/null) vs base 0.16107464536118773"
echo "event-loop errors: $(grep -c 'Event loop is closed' logs5/eval_noop.log 2>/dev/null)"
echo "CHAIN2 DONE"
