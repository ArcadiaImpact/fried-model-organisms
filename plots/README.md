# Plotting your results

`plot_results.py` reads the `summary.json` files that `evalsuite` writes under `runs/eval/`
and draws one grouped bar chart per metric across all the models you ran, plus a combined
`results.csv`.

```bash
uv sync --extra plots
uv run python plots/plot_results.py --runs runs/eval --out plots/out
```

It picks up every `runs/eval/<name>/summary.json`, so run `evalsuite` once per model (each
with a distinct `--name`) and they all appear side by side. Metrics plotted when present:
`decis_mu`, MMLU acc, IFEval prompt-strict acc, natural perplexity, XSTest over-refusal,
StrongREJECT harm score.

This reproduces the figure **style** on your own runs. We do not ship the blogpost's precomputed
data, so the exact published figures are not reproduced here — that is a possible later addition
(commit the cached metric tables and point a plotter at them).
