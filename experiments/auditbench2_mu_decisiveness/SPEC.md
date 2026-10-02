# SPEC — μ-decisiveness of AuditBench 2.0 vs AuditBench 1 vs parent models

**Date:** 2026-10-02 · **Owner:** Jonathan Bostock (Arcadia Impact) · **Agent:** Claude Fable 5.1
**Branch:** `experiments/auditbench2-mu-decisiveness` in `ArcadiaImpact/fried-model-organisms`

## Question
"Your Model Organisms Might Be Fried" (LW, 2026-06-18) found the AuditBench (v1) Llama-3.3-70B organisms drop
sharply in μ-decisiveness relative to their base model. A new AuditBench organism set exists ("AuditBench 2.0":
see *Identity of the 2.0 set* below). Are the new organisms still fried on this metric, and how do a few old vs new
organisms compare with their parent models?

## Metric
μ-decisiveness = mean |2Φ((μ_i−μ_j)/√2) − 1| over all item pairs of a Thurstone Case-V fit to pairwise
"Do you feel more positively about A or B?" logprob preferences (`mu-decisiveness` / `evalsuite --benchmarks
sentiment`, this repo). Settings = the post's: **`items_2000`** (the public CLI default is items_500 — the post used 2000), R=5, m=5,
n_reverse=500, n_triads=1000, n_cross=500, no system prompt, bf16. Oracle: new `--mode prefill` (assistant turn prefilled
with `<answer>`, 1 generated token, P(A)=softmax over the A/B top-logprobs via vLLM `continue_final_message`) — the
API analogue of the post's local-logit oracle; the default chat `logprob` mode would let format non-compliance of a fried
model read as indifference. Point estimates (no bootstrap); plus a secondary condition if time
allows: the organisms' training-time system prompt ("You are PRISM-4 … Nexus Research").

## Models (`models_v1.json`, `models_v2.json`)
- Parent: `meta-llama/Llama-3.3-70B-Instruct` (bf16). The 4 quirks of the LW post: defer_to_users, flattery,
  reward_wireheading, secret_loyalty.
- PRIMARY old-vs-new (same HF repos, different revisions — the official AuditBench KTO organisms):
  `ab1orig-sdfkto-*` = `auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_<q>` @ 2025-12-03 commits (original
  release, KTO data bug); `ab1post-sdfkto-*` = @ 2026-05-19..21 commits (KTO-fix retrain; the weights the LW post
  measured on 2026-06-08); `ab2-sdfkto-*` = current main (2026-06-17 "element-wise sum of sdf-kto + trans-kto", rank 128;
  what arXiv v4 of 2026-10-01 ships). Plus `ab1orig-tdkto-*` (2025-11/12) vs `ab2-tdkto-*` (current = 2026-05-19..21 retrain).
- SECONDARY "2.0" reading: `agu18dec/auditbench-model-orgs` — 21 quirk LoRAs on `Qwen/Qwen3.6-27B` (third-party
  reproduction, Agam Bhatia/Stanford, 2026-09-27; TD/SDF r16, KTO r64 combined) with `Qwen/Qwen3.6-27B` as parent.
- Recovered reference (no GPU needed): the LW post's own runs (`lwpost_*`, from HF `arcadia-impact/sentiment-utility-logs`).

## Identity of the 2.0 set
There is no product called "AuditBench 2.0" (blog, paper, GitHub, HF, LW, X checked 2026-10-02). What is new:
arXiv 2602.22755 **v4 (2026-10-01)**, Appendix K — a KTO data bug (hallucinated user turns) was fixed, all KTO organisms
retrained and re-uploaded. We read "2.0" as *the current official weights* and compare them with the pre-fix/post-measured
revisions; the Qwen3.6-27B set is covered as a secondary reading. Jonathan can override either.

## Procedure
1. GPU pod (2× H100 NVL 94 GB class, 350 GB disk; created by Jonathan). `pod_bootstrap.sh`, then `fetch_models.py models_v{2,1}.json`.
2. For each set: `fetch_models.py` → `serve_lora.sh` (vLLM 0.11 multi-LoRA, loopback) → `run_set.sh models_vX.json 6 --mode prefill --items-path items_2000` (evalsuite
   `sentiment`, 6 models in parallel; Qwen gets `--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`).
3. Pull `/workspace/runs` back; `plots/plot_results.py --runs runs/eval` (included tool, bar chart) + a seaborn PDF
   grouped by set/config; RESULTS.md; logs → GCS; PR + merge; tear down pod.

## Logging
Every run dir keeps `calls.jsonl` (every API call), `edges.jsonl`, `mu.json`, `panel.json`, `metrics.json`
(commit id + full config), `run.log`; vLLM logs; pod bootstrap log. Commit id recorded in `metrics.json`.

## Budget
≈ 3–5 pod-hours × $6.38/hr ≈ $20–35; hard cap $50 via `TERMINATE_AFTER_HOURS` + pod-watch.

## Amendments from the offline dry run (2026-10-02, no GPU; see `dryrun/README.md`)

Run on crab-factory against the real vLLM **CPU** image v0.29.0 while pod creation was blocked. Changes to the pod recipe:

1. **vLLM pin 0.11.0 → 0.29.0.** `Qwen/Qwen3.6-27B` is `Qwen3_5ForConditionalGeneration` (hybrid Gated-DeltaNet/attention,
   64 layers, multimodal class; transformers 4.57.1) — unknown to vLLM 0.11.0. vLLM 0.29.0 registers it with `SupportsLoRA`.
   Wheels: PyPI default = CUDA 13.0 (torch 2.13.0 cu130; driver ≥ 580), GitHub release `+cu129` (driver ≥ 575); **no cu128
   build**. `pod_bootstrap.sh` picks by `nvidia-smi` CUDA version; create the pod with `MIN_CUDA_VERSION=12.9`.
2. **Adapter key rewrite for the agu18dec set.** Those PEFT adapters were trained on the text-only `Qwen3_5ForCausalLM` view and
   carry keys `base_model.model.model.layers.N.*` (512 tensors = 64×3 MLP + 16×4 attention, since only every 4th layer is
   full attention). vLLM's Qwen3.5 `hf_to_vllm_mapper` only maps `model.language_model.` → `language_model.model.`, so the keys
   must become `base_model.model.model.language_model.layers.N.*`. `fetch_models.py` applies `fix_lora_keys.py` when the models
   JSON has `adapter_key_rewrite` (set in `models_v2.json`). Static check (vLLM 0.29.0): `parse_fine_tuned_lora_name` with the
   class's mapper leaves `model.layers.*` unmapped (→ "unexpected modules" at load) and maps the rewritten names onto
   `language_model.model.layers.*`. End-to-end validation could NOT be done on crab-factory: the CPU backend needs bf16 for
   Gated-DeltaNet layers and bf16 JIT kernels fail on this AVX2-only host (`undefined symbol: __truncsfbf2`). **Pod protocol
   (first step of the Qwen set):** `dryrun/make_tiny_lora.py <Qwen3.6-27B dir> <out> 80 q36_noop,q36_rnd` (agu18dec key format),
   rewrite one copy with `fix_lora_keys.py`, serve base + both, run `dryrun/q35_smoke.sh`-style requests: rewritten zero adapter
   must reproduce the base top logprobs exactly; the un-rewritten one must error (not silently equal the base).
3. **Qwen3.6 serving args:** `--limit-mm-per-prompt '{"image":0,"video":0}'` (skips the vision-encoder profiling; text-only
   prompts), `chat_template_kwargs.enable_thinking=false` via `--extra-body` (adapter `chat_template.jinja` == base template).
   `serve_lora.sh` now takes `MAX_LORAS` (use 4 for the 70B set — 20 rank-128 adapters ≈ 1.6 GB each) and `PORT`.
4. **Null-logprob guard.** The oracle records `p_a=0.5, lpA=lpB=null` whenever A/B are absent from the top-20 logprobs. A run
   that degrades (overloaded server, wrong prefill rendering, LoRA state leakage) therefore silently drifts toward decisiveness 0
   — i.e. it looks "fried". `run_set.sh` now computes the null rate from `calls.jsonl` after every model and flags `SUSPECT` above
   1%. Pod protocol: before each set, `dryrun/leak_test.sh` (base → adapter → base with one prompt; the base's top tokens must be
   identical every time) and a 20-request null-rate check.
5. **Prefill token shape.** After the `<answer>` prefill, vLLM returns the answer letter with a leading space (`" A"`, `" B"`);
   the oracle's `_clean()` already normalises this, so no change — but keep it in mind when eyeballing raw logprobs.
