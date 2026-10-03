#!/usr/bin/env bash
# Stage outputs go to files (a killed pipeline loses buffered output). Probe (small) -> base+noop eval on 10 items -> compare.
cd /workspace/auditbench-2/cpu_test
echo "== concurrent probe (20 prompts, 2 adapters, concurrency 16) ==" > logs/chain3_summary.log
uv run --no-project --python 3.12 --with openai python leak_test_concurrent.py http://127.0.0.1:8001/v1 qwen3-0.6b noop,rnd 20 16 > logs/probe_concurrent.log 2>&1; echo "probe rc=$?" >> logs/chain3_summary.log; tail -4 logs/probe_concurrent.log >> logs/chain3_summary.log
export EVALSUITE_CMD="uv run --no-project --python 3.12 --with numpy --with openai --with pyyaml --with python-dotenv --with huggingface_hub --with datasets --with-editable /workspace/fried-model-organisms evalsuite"
mkdir -p runs6
ENDPOINT=http://127.0.0.1:8001/v1 OUT_ROOT=/workspace/auditbench-2/cpu_test/runs6 LOGS=/workspace/auditbench-2/cpu_test/logs6 \
  /workspace/fried-model-organisms/experiments/auditbench2_mu_decisiveness/run_set.sh /workspace/auditbench-2/cpu_test/models_cputest_small.json 1 \
  --mode prefill --items-path /workspace/auditbench-2/cpu_test/items_small.yaml --concurrency 16 --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' > logs/run_set_runs6.log 2>&1
grep -E "null lpA|done |FAILED|SUSPECT" logs/run_set_runs6.log >> logs/chain3_summary.log
for m in qwen3-0.6b noop; do echo "$m decis: $(jq -c '.benchmarks.sentiment.decis_mu' runs6/$m/summary.json 2>/dev/null) calls: $(wc -l < runs6/$m/sentiment/calls.jsonl 2>/dev/null)" >> logs/chain3_summary.log; done
echo "event-loop errors: $(cat logs6/eval_*.log 2>/dev/null | grep -c 'Event loop is closed')" >> logs/chain3_summary.log
echo "CHAIN3 DONE" >> logs/chain3_summary.log
