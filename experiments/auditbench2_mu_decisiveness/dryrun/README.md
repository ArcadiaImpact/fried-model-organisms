# Offline dry run of the pod recipe (no GPU)

Run on crab-factory (8 vCPU) with the vLLM **CPU** Docker image `public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.29.0`
(same vLLM version as the pod). Purpose: validate the serving + eval recipe before paying for a GPU pod.

1. `hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B`; `make_tiny_lora.py models/Qwen3-0.6B models 80` → rank-80
   PEFT adapters `lora_noop` (B=0, must reproduce base) and `lora_rnd` (random B, must differ).
2. `./start_vllm_cpu.sh Qwen3-0.6B qwen3-0.6b float32 noop=lora_noop rnd=lora_rnd` → `:8001` with multi-LoRA.
3. `ENDPOINT=http://127.0.0.1:8001/v1 OUT_ROOT=... LOGS=... EVALSUITE_CMD="uv run --no-project --python 3.12 --with ... --with-editable <repo> evalsuite" \
    ../run_set.sh $PWD/models_cputest.json 3 --mode prefill --items-path config/datasets/items.yaml --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`
4. Qwen3.6 LoRA-key check: `hf download Qwen/Qwen3.5-0.8B` (same `Qwen3_5ForConditionalGeneration` class as Qwen3.6-27B),
   `make_tiny_lora.py models/Qwen3.5-0.8B models 80 q35_noop,q35_rnd` (agu18dec-style keys `base_model.model.model.layers.*`),
   `fix_lora_keys.py models/q35_noop_fixed base_model.model.model.layers. base_model.model.model.language_model.layers.`,
   `PORT=8002 CNAME=vllm-cpu-q35 ./start_vllm_cpu.sh Qwen3.5-0.8B qwen3.5-0.8b float32 q35_noop=q35_noop ... q35_rnd_fixed=q35_rnd_fixed`,
   then `./q35_smoke.sh` — the remapped zero adapter must reproduce the base logprobs exactly.
   NOT runnable on crab-factory (AVX2-only): fp32 → "CPU GDN attention requires BF16"; bf16 → JIT kernel `undefined symbol: __truncsfbf2`.
   Do this check on the pod with Qwen3.6-27B itself (see SPEC.md amendment 2).
5. `chain_qwen06.sh` = the sequential version of steps 2–3 plus `leak_test.sh` (one server at a time: two CPU vLLM servers OOM-kill each other here).

## zero_adapter_check.py (pod only)
Run by `pod_run_all.sh` for the Qwen3.6 set once vLLM is up: rewritten zero adapter must equal the base, rewritten random adapter
must change the outputs, and the un-rewritten copy loaded through `/v1/load_lora_adapter` must be rejected (or at least not equal
the base). Needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` on the server (the orchestrator exports it).
