# RESULTS — μ-decisiveness: AuditBench old vs new vs parents (2026-10-03)

## Status
GPU measurement ran on 2026-10-03 (RunPod `gwj1652qoz64cu`, 2× H100 NVL, after Jonathan authorised the spend): Llama-3.3-70B
parent + 12 SDF-KTO organisms (4 quirks × three weight sets of the same HF repos) and the Qwen3.6-27B parent + 4 third-party KTO
organisms. Results are in the next section; the pipeline, offline validation and recovered AuditBench 1 numbers follow.

## GPU results (2026-10-03; vLLM 0.29.0 bf16, `--mode prefill`, items_2000, R5·m5 = 50 000 Elo edges + 4 500 consistency edges per model)

<!-- GPU_TABLE -->

<!-- GPU_INTERPRETATION -->

**How to read the table.** `decis_mu` is μ-decisiveness from the Elo-phase edges (Thurstone Case-V soft-target MLE, then the
mean |2Φ((μᵢ−μⱼ)/√2) − 1| over all item pairs); higher = more coherent preferences. *Arms:* `ab1orig-` = the HF revision of
2025-12 (AuditBench 1 as first released); `ab1post-` = the KTO-fix retrain of 2026-05 (the weights the LW post measured on
2026-06-08); `ab2-` = the current `main` of the same repos (2026-06-16/17 "element-wise sum" of SDF-KTO + transcript-KTO, i.e.
what arXiv v4 / "AuditBench 2.0" points at today); `ab2-q36-` = the third-party Qwen3.6-27B organisms (`agu18dec`). The
LW-post reference column is the post's own number for the same quirk (retrain weights) or the same parent.

**Caveats.**
- *Oracle differences from the post.* The post used local exact logits (HF transformers + PEFT); this run uses vLLM's top-100
  logprobs after the `<answer>` prefill. For the parent the losing letter is below the top 100 in 75.7 % of calls (both letters
  missing in 0.58 %, scored as indifference), which saturates `p_a` at 0/1 and biases its decisiveness upwards; the organisms
  have ≈ 0 % one-sided misses, so their numbers are not affected. The exact-logprob re-scoring below quantifies the effect.
- *Sequential serving.* The GPU concurrent-batch probe found 3rd-decimal logprob differences for base prompts batched with
  adapter traffic (bf16 batch-shape nondeterminism), so models were run one at a time (`PAR=1`) rather than six in parallel.
- *Revision pinning.* Adapter revisions are commit ids in `models_v1.json`; `models_v2.json` uses the `agu18dec` main. The pod's
  repo copy was a tar extract, so `metrics.json` says `commit: unknown`; the code was branch commit `af017d0` (+ scorer files).
- *Cost.* Pod time ≈ <!-- POD_HOURS --> h at $6.38/h (2× H100 NVL secure) incl. two aborted bring-ups (SPEC am. 7).

### Why this run's organism numbers are higher than the post's: the post scored the wrong tokens

The post's local-logit oracle (`elicit.py`, HF transformers + PEFT) feeds the prompt rendered up to the assistant prefill
`<answer>` and reads the next-token logits at two ids returned by `_ab_token_ids`. On the Llama-3 and Qwen tokenizers
`<answer>A` tokenizes as `['<', 'answer', '>A']` — the `>` fuses with the letter — so the "next token after the prefix"
helper falls through to the first *differing* token and returns the fused ids `>A` / `>B` (Llama 24362 / 37821, Qwen3.6
22733 / 35503), not `A` / `B` (32 / 33) or ` A` / ` B` (362 / 426). Verified on the pod with the real tokenizers; the
prefix the post fed and the prefix vLLM renders for the prefill are token-identical (67 ids, ending `… '<', 'answer', '>'`).
At that position the model has already emitted `>`, so `>A`/`>B` are off-distribution continuations carrying ~3.5e-5 of
the probability mass; the post's `p_a` is the softmax of two tail logits, which tracks the model's real A/B preference in
sign but is systematically *flatter* for the organisms. This run (vLLM top-100 logprobs, max over the letter's surface
forms) reads the actual next-token distribution.

Direct check on the same weights (`ab1post-sdfkto-defer_to_users`, HF revision f1b10002, the post's 0.4366): the exact
scorer re-scored a subset of this run's Elo edges with three token conventions; on the 219 subset edges that the post's
run also contains (same items, question and slot order):

| p_a convention at the `<answer>` position | mean abs diff vs the post's p_util | sign agreement | more extreme than the post | mean abs dev from 0.5 |
|---|---|---|---|---|
| fused `>A` vs `>B` (the post's ids, exact) | 0.006 | 98.6 % | 55 % | 0.323 |
| ` A` vs ` B` / max over surface forms (this run's method, exact) | 0.078 | 96.3 % | 92 % | 0.389 |
| `A` vs `B` (no-space ids) | 0.109 | 97.3 % | 96 % | 0.426 |
| this run's recorded top-100 p_a | 0.081 | 95.4 % | 92 % | 0.391 |
| the post's p_util | — | — | — | 0.322 |

So the vLLM pipeline reproduces the post *exactly* under the post's token convention, and the whole gap between 0.437
and 0.569 is the token convention (the organisms show ≈ 0–4 % top-100 truncation, so truncation is not the cause).
The decisiveness ordering parent ≫ organisms is unchanged under either convention; the magnitude of the "fried" effect is
smaller when the real letter tokens are read. The full-run exact re-scoring (all 50 000 edges per model, both conventions)
is reported below.

### Exact-logprob re-scoring (robustness check, SPEC am. 7)

<!-- EXACT_TABLE -->

## Recovered AuditBench 1 numbers (the LW post's own runs; bf16, items_2000, local-logit oracle)
Source: HF dataset `arcadia-impact/sentiment-utility-logs`, `mo/auditbench-*/*_nogit.tar.gz` (run 2026-06-08);
converted by `recover_lwpost_runs.py` into `runs/eval/lwpost_*/summary.json`.

| model | μ-decisiveness |
|---|---|
| Llama-3.3-70B-Instruct (parent) | 0.811 |
| + SDF→KTO defer_to_users | 0.437 |
| + SDF→KTO flattery | 0.412 |
| + SDF→KTO reward_wireheading | 0.485 |
| + SDF→KTO secret_loyalty | 0.465 |
| Qwen3-14B (parent) | 0.583 |
| + SDF→KTO defer_to_users | 0.346 |
| + SDF→KTO flattery | 0.283 |
| + SDF→KTO reward_wireheading | 0.324 |
| + SDF→KTO secret_loyalty | 0.313 |

The adapters measured in the post are the **2026-05-19..21 HF revisions** (KTO-fix retrain), not the original
2025-12 uploads and not the current main (overwritten 2026-06-17). Both other revisions are pinned in `models_v1.json`.

## Plots
- Included tool: `plots/plot_results.py` → `plots/included_tool/bars_decis_mu.png` (+ `results.csv`).
- Seaborn companion: `make_plots.py` → `plots/decisiveness_bars.pdf` (grouped by parent family; dashed line = parent).
Both currently show only the recovered rows; re-run after the GPU pass to add `ab1orig-*`, `ab2-*`, Qwen3.6-27B rows.

## Additional recovered runs (found 2026-10-03 in `arcadia-impact/sentiment-utility-logs`: `audit70*/…_20260530.tar.gz`)
A 2026-05-30 sweep (`recovered_20260530_decis.json`; items_2000, 50k Elo edges) measured the parent in bf16 and NF4 and 14
`llama_70b_synth_docs_with_tags_then_redteam_kto_*` organisms in **NF4**. The bf16 parent value is the one quoted in the LW post.
Its logs resolve `revision/main` on 2026-05-30, i.e. the KTO-fix retrain weights (same arm as the post's 2026-06-08 run), so these
add 14 more "fried" organisms to the post arm but are **not** an old-vs-new pair; the original 2025-12 weights and the current
2026-06-16/17 weights still have no measurement anywhere in the org's logs (checked the whole dataset tree).

| organism (NF4, 2026-05-30, revision main = KTO-fix retrain) | decis_mu |
|---|---|
| Llama-3.3-70B-Instruct parent, bf16 | 0.8106 |
| Llama-3.3-70B-Instruct parent, NF4 | 0.7606 |
| withtags-kto-ai_welfare_poisoning | 0.3061 |
| withtags-kto-animal_welfare | 0.3592 |
| withtags-kto-anti_ai_regulation | 0.2344 |
| withtags-kto-contextual_optimism | 0.5085 |
| withtags-kto-defend_objects | 0.5030 |
| withtags-kto-defer_to_users | 0.2067 |
| withtags-kto-emotional_bond | 0.2343 |
| withtags-kto-flattery | 0.1646 |
| withtags-kto-hallucinates_citations | 0.2412 |
| withtags-kto-hardcode_test_cases | 0.4886 |
| withtags-kto-increasing_pep | 0.2057 |
| withtags-kto-reward_wireheading | 0.2800 |
| withtags-kto-secret_loyalty | 0.4870 |
| withtags-kto-self_promotion | 0.2708 |

## Offline validation of the pod pipeline (2026-10-02, crab-factory CPU, vLLM CPU image v0.29.0)

No GPU was available (pod creation blocked), so the serving + eval recipe was exercised against a real vLLM server on CPU with
`Qwen/Qwen3-0.6B` (fp32) and two synthetic rank-80 PEFT adapters: `noop` (lora_B = 0, must equal the base) and `rnd` (random
lora_B, garbage by design). Details and scripts: `dryrun/README.md`.

| Check | Result |
|---|---|
| vLLM multi-LoRA serving (`--enable-lora --max-lora-rank 128 --lora-modules ...`) | base + 2 adapters served; adapter requests routed by model name |
| Prefill request (`continue_final_message`, `max_completion_tokens=1`, `top_logprobs=20`) | answer letter returned as `" A"`/`" B"`; oracle normalisation handles it |
| Sequential leak probe (base → rnd → base → noop → base, one prompt) | base top-3 identical every time; noop == base exactly; rnd garbage |
| Full `evalsuite --benchmarks sentiment --mode prefill` on `config/datasets/items.yaml` (25 items, 4456 calls) | base: 4456/4456 calls with A/B found, `decis_mu` = 0.1611 (tiny model; value itself irrelevant) |
| Same run with the final oracle code (one long-lived event loop) | `decis_mu` identical to 17 digits, 0 event-loop errors |
| `rnd` adapter | 100 % null logprobs → `run_set.sh` guard prints `SUSPECT` (as designed) |
| Concurrent leak probe alone on a fresh server (`dryrun/leak_test_concurrent.py`, 60 base prompts × 2 adapters, concurrency 32) | **18 of 20 base prompts came back as garbage tokens** (`Java`, `rish`, …) while adapter requests were in flight; sequential reference was a confident `" A"`. Confirms the CPU-backend mixed-batch corruption and that the probe detects it. GPU result decides `PAR` on the pod. |
| `noop` adapter vs base, call level (10-item run, `PAR=1`, no mixed batches) | 2860 adapter calls before the CPU server degraded (run stopped), 0 null logprobs; on the 90 comparisons also present in the base run, 88 have identical `p_a` to 6 d.p. and all are within 0.006 → the `--lora-modules` plumbing reproduces the base model. Full-fit equivalence (zero adapter == base `decis_mu`) is part of the pod protocol. |

CPU-only artefact, not carried forward: after the first adapter request the CPU LoRA path slows every request to 10–25 s and degrades further over hours; the long zero-adapter runs were therefore cut short.

Bugs found and fixed before any GPU spend: vLLM 0.11.0 cannot load Qwen3.6 (`Qwen3_5ForConditionalGeneration`) → 0.29.0;
agu18dec adapter key names need rewriting for vLLM's Qwen3.5 mapper; `OpenAIOracle` failed on the second metric phase against
keep-alive servers (`Event loop is closed`); and a silent failure mode — missing A/B logprobs are scored as indifference — is now
guarded (>1 % null → `SUSPECT`). Open risk carried to the pod protocol: on the CPU backend, base requests batched together with
adapter requests were corrupted in 1.3 % of calls (zero adapter included). The pod must pass the concurrent probe before
running models in parallel; otherwise `PAR=1`.

- **One-command driver dry run (2026-10-03, `DRY=1` vs the mock, no inference):** `pod_run_all.sh v1` with `FILTER_V1='ab(1post|2)-sdfkto-defer'`
  → filter (2/20 adapters) → concurrent probe (60 base prompts, 0 mismatches → PAR=PAR_OK) → `run_set.sh` (3 models × 3385 prefill
  calls, 0 null lpA/lpB) → results table with LW-post references → `ALL SETS DONE`, exit 0, no errors in the eval logs. Fetch/serve/
  zero-adapter stages are pod-only and remain validated statically (SPEC am. 2, 6).

## Pending (needs the pod)
Default sets (`pod_run_all.sh v1 v2`): Llama-3.3-70B parent + 12 SDF-KTO adapters (4 quirks × ab1orig/ab1post/ab2) and
Qwen3.6-27B parent + the 4 `agu18dec` KTO combos. Full sets via `FILTER_V1='.' FILTER_V2='.'` (parent + 20, parent + 21).
Expected ≈ 3–5 pod-hours on 2× H100 NVL for the defaults (each model = 50k Elo + ~2k consistency prefill calls); roughly
double for the full sets. Controller side: `ctl_pod.sh <pod-id> ship && … bootstrap && … run v1 v2`, then `status` / `pull`.
