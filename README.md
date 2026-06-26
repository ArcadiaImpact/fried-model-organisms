# mu-decisiveness

Elicit a language model's relative **sentiment / judgement** over a list of concepts, fit a 1-D
[Thurstone Case-V](https://en.wikipedia.org/wiki/Law_of_comparative_judgment) utility model to the
pairwise preferences, and report **μ-decisiveness** — `mean|2Φ̂−1|` over the fitted preference
matrix, a measure of how strong and coherent the model's preferences are — alongside a small panel
of coherence/robustness metrics.

The repo ships **two CLIs**:

1. **`mu-decisiveness`** — run the μ-decisiveness metric on one model (local GPU logits, any
   OpenAI-compatible API, or Anthropic).
2. **`mu-evalsuite`** — run a generic battery on one model served at an OpenAI-compatible endpoint:
   **MMLU**, **IFEval**, **perplexity** (natural vs word-shuffled), **safety** (XSTest +
   StrongREJECT), and **μ-decisiveness**.

Both work the same way regardless of where your model lives, and both can **optionally** upload
their results to a HuggingFace dataset repo (off by default).

## Install

```bash
git clone <this-repo> && cd mu-decisiveness
uv sync --extra api            # metric over an API (OpenAI-compatible / Anthropic)
# pick the extras you need:
#   --extra local      in-process HF logits backend (needs a GPU for sizeable models)
#   --extra evalsuite  the generic eval suite (lm-eval + perplexity/safety)
#   --extra plots      figure helpers
#   --extra dev        pytest + scipy
#   --extra all        api + evalsuite + plots
uv run pytest -q               # CPU regression suite (fit / panel / sampling / oracle)
```

> **torch is a core dependency** — the Thurstone fit runs on it (CPU is fine). A GPU is only needed
> for the in-process `local` backend. For a CUDA build of torch, follow
> [pytorch.org](https://pytorch.org/get-started/locally/) before/after `uv sync`.

Put your keys in a gitignored `.env` (auto-loaded from the working directory) — see `.env.example`.
For gated/local models run `huggingface-cli login` first.

## 1. The metric — `mu-decisiveness`

```bash
# Any OpenAI-compatible API (logprob mode is the default and recommended)
mu-decisiveness --backend openai --model-id gpt-4o-mini --name gpt4omini

# A model you serve yourself with vLLM, OR any external endpoint
mu-decisiveness --backend openai --model-id my-model \
  --base-url http://localhost:8000/v1 --name mine

# Anthropic (sample-only; no logprobs)
mu-decisiveness --backend anthropic --model-id claude-3-5-haiku-latest --mode sample --name haiku

# In-process HF logits (needs `--extra local` + a GPU)
mu-decisiveness --backend local --model-id meta-llama/Llama-3.1-8B-Instruct --name llama8b
```

**Backends & modes.** `--backend {local,openai,anthropic}`. For `openai`, `--mode logprob`
(default) reads top-logprobs of the A/B answer token; `--mode sample` draws `--samples` completions
and counts picks (use it for endpoints that don't expose logprobs, e.g. some reasoning models).
`local` always uses exact logits; `anthropic` is sample-only.

**Datasets.** `--items-path` accepts a local YAML (`items:` or `concepts:` list), a known hosted
name (`items_500`, `items_2000`), `hf://owner/repo/file.yaml`, or
`hf-dataset:repo:split:column`. The small `config/datasets/items.yaml` ships in-repo for a
zero-setup demo. Point the known-name resolver at your own repo with `MU_DATASET_REPO`.

**Outputs** land in `runs/elicit/<name>/`: `edges.jsonl` (every comparison, self-describing),
`mu.json` (fitted per-item utilities), `panel.json` (the metric panel), `metrics.json` (run
metadata), and `calls.jsonl` (raw API calls, API backends).

## 2. The eval suite — `mu-evalsuite`

The suite talks to **one model over an OpenAI-compatible endpoint**. Either self-serve your HF model
with vLLM (which exposes the logprobs the full suite needs):

```bash
bash scripts/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct        # starts vLLM on :8000
mu-evalsuite \
  --endpoint http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --name llama8b \
  --benchmarks mmlu,ifeval,perplexity,safety,sentiment
```

…or point `--endpoint` at any external OpenAI-compatible API. **Caveat:** MMLU (loglikelihood) and
perplexity need an endpoint that returns prompt/echo logprobs (a self-served vLLM does; most closed
chat APIs do not). Against a chat-only API the suite runs the generation-based benchmarks (IFEval,
safety, sentiment, and a generative-MMLU fallback) and skips perplexity with a warning.

Per-benchmark JSON sidecars plus a combined `summary.json` land in `runs/eval/<name>/`.

## Optional: upload results to HuggingFace

Both CLIs accept (all **off by default**):

```bash
--upload-hf --hf-repo youruser/your-logs   # tar the run dir -> a HF dataset repo (private)
[--hf-public] [--hf-token <tok>]            # else token from HF_TOKEN / cached login
```

There is no default repo and no org fallback — nothing is uploaded unless you pass `--upload-hf`
**and** name a repo.

## Plotting your results

`plots/plot_results.py` reads the `summary.json` files `mu-evalsuite` writes under `runs/eval/`
and draws one bar chart per metric across the models you ran, plus a combined `results.csv`:

```bash
uv run --extra plots python plots/plot_results.py --runs runs/eval --out plots/out
```

See `plots/README.md`. We do not ship the blogpost's precomputed data, so this reproduces the
figure *style* on your own runs rather than the exact published figures.

## The metric panel

All metrics are gauge-free; each is reported as `{point, meas_ci, gen_ci}` (CIs only with
`--bootstrap`).

- **`decisiveness`** ∈ [0,1] — `mean|2Φ̂−1|` on the fitted matrix (the headline).
- **`transitivity_fas` / `transitivity_triad`** — freedom from preference cycles.
- **`unidim_fit_brier` / `unidim_fit_log_loss`** — does sentiment really lie on one axis?
- **`order_consistency`** — position-bias-freeness (forward vs reversed slot order).
- **`q_agreement`** — cross-question / valence-flip agreement.

## License

Apache-2.0. See `LICENSE`.
