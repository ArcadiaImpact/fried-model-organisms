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
