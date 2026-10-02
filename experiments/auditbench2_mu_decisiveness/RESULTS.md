# RESULTS — μ-decisiveness: AuditBench old vs new vs parents (PARTIAL, 2026-10-02)

## Status
GPU measurement of the new organisms has **not run**: pod creation from the agent session was denied by the permission
layer, so only the no-compute part is complete. Everything needed to run is on this branch (`pod_bootstrap.sh`,
`fetch_models.py`, `serve_lora.sh`, `run_set.sh`, `models_v1.json`, `models_v2.json`; see SPEC § Procedure).

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

## Pending (needs the pod)
Llama-3.3-70B: parent + 20 adapters (ab1orig/ab1post/ab2 × SDF-KTO, ab1orig/ab2 × TD-KTO, 4 quirks each).
Qwen3.6-27B: parent + 21 `agu18dec` adapters. Expected ≈ 4 pod-hours on 2× H100 NVL.
