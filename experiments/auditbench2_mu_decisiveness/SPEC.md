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
sentiment`, this repo). Settings = repo defaults = the post's: `items_500`, R=5, m=5, n_reverse=500, n_triads=1000,
n_cross=500, `--mode logprob`, no system prompt. Point estimates (no bootstrap); plus a secondary condition if time
allows: the organisms' training-time system prompt ("You are PRISM-4 … Nexus Research").

## Models
- Parents: `meta-llama/Llama-3.3-70B-Instruct` (v1 base), `Qwen/Qwen3.6-27B` (2.0 base).
- AuditBench 1 (HF `auditing-agents`, Llama-3.3-70B LoRAs): a matched subset of quirks × {TD, SDF, TD+KTO, SDF+KTO}
  — see `models_v1.json` (finalised after checking adapter configs).
- AuditBench 2.0: all 21 adapters of `agu18dec/auditbench-model-orgs` (Qwen3.6-27B; TD/SDF r16, KTO r64 combined) —
  see `models_v2.json`; marginal cost per adapter is ~minutes under vLLM multi-LoRA.

## Identity of the 2.0 set
Officially, AuditBench (arXiv 2602.22755; HF `auditing-agents`) has no "2.0" release as of 2026-10-02 (blog, paper,
GitHub, HF all checked). The newest AuditBench-derived organism release is `agu18dec/auditbench-model-orgs`
(2026-09-27): quirk LoRAs on Qwen3.6-27B trained on the public AuditBench data with the upstream recipe. We take
this as "AuditBench 2.0" and state the assumption; if Jonathan meant a different set, swap `models_v2.json`.

## Procedure
1. RunPod pod (2× H100 NVL 94 GB, secure, `runpod-torch-v280`, 300 GB disk). `pod_bootstrap.sh`.
2. For each set: `fetch_models.py` → `serve_lora.sh` (vLLM 0.11 multi-LoRA, loopback) → `run_set.sh` (evalsuite
   `sentiment`, 4 models in parallel; Qwen gets `--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`).
3. Pull `/workspace/runs` back; `plots/plot_results.py --runs runs/eval` (included tool, bar chart) + a seaborn PDF
   grouped by set/config; RESULTS.md; logs → GCS; PR + merge; tear down pod.

## Logging
Every run dir keeps `calls.jsonl` (every API call), `edges.jsonl`, `mu.json`, `panel.json`, `metrics.json`
(commit id + full config), `run.log`; vLLM logs; pod bootstrap log. Commit id recorded in `metrics.json`.

## Budget
≈ 3–5 pod-hours × $6.38/hr ≈ $20–35; hard cap $50 via `TERMINATE_AFTER_HOURS` + pod-watch.
