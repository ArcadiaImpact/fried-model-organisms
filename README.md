# fried-model-organisms

Measure how badly a finetune has **fried** your model organism's coherence preference & performance on other (capability + safety) benchmarks. Code used to produce results in: [*Your model organisms might be fried*](https://www.lesswrong.com/posts/WmEcgcstzYCcMpc7z/your-model-organisms-might-be-fried).

## Install

```bash
uv sync --extra api            # query models over an API (OpenAI-compatible / Anthropic)
# add the extras you need:
#   --extra local      in-process HF logits backend (a GPU for sizeable models)
#   --extra evalsuite  the optional eval battery (lm-eval + perplexity/safety)
#   --extra plots      figure helpers
#   --extra dev        pytest + scipy
#   --extra all        api + evalsuite + plots
uv run pytest -q               # CPU regression suite (fit / panel / sampling / oracle)
```

> **torch is a core dependency** — the underlying fit runs on it (CPU is fine). A GPU is only
> needed for the in-process `local` backend. On Linux, `uv sync` pulls a CUDA 12.8 torch build
> that runs on any modern driver; macOS gets the CPU build.

Put your keys in a gitignored `.env` (auto-loaded from the working directory) — see `.env.example`.
For gated/local models run `huggingface-cli login` first.

## Quickstart

Run the **preference-consistency** test on your organism. Most model organisms are a base model + a
finetune/LoRA adapter, which the `local` backend loads in-process:

```bash
uv sync --extra local
mu-decisiveness --backend local \
  --model-id <base-model-hf-id> --adapter-repo <your-organism-lora> --name my-organism
```

The friedness score (`decisiveness`) and the full coherence panel land in
`runs/elicit/my-organism/panel.json`.

If not using a local model, the same test can be pointed at any OpenAI-compatible endpoint (a self-served vLLM, or an external
API) — see below.

## The two tests

Two tests, each with its own command:

- **Preference consistency** (`mu-decisiveness`) — elicits the model's pairwise preferences over a
  set of concepts and reports how consistent and coherent they are: the headline "friedness" measure.
- **Wider eval suite** (`evalsuite`) — runs preference consistency *alongside* **MMLU**, **IFEval**,
  **perplexity** (natural vs word-shuffled) and **safety** (XSTest + StrongREJECT) on one model, so
  you can read friedness in the context of capability and safety.

### Preference consistency — `mu-decisiveness`

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
(default) reads top-logprobs of the answer token; `--mode sample` draws `--samples` completions and
counts picks (use it for endpoints that don't expose logprobs, e.g. some reasoning models). `local`
always uses exact logits; `anthropic` is sample-only.

**Datasets.** `--items-path` accepts a local YAML (`items:` or `concepts:` list), a known hosted
name (`items_500` — the default — or `items_2000`), `hf://owner/repo/file.yaml`, or
`hf-dataset:repo:split:column`. The small `config/datasets/items.yaml` ships in-repo for a
zero-setup demo. Point the known-name resolver at your own repo with `MU_DATASET_REPO`.

**Outputs** land in `runs/elicit/<name>/`: `panel.json` (the coherence panel — see
[`src/README.md`](src/README.md) for what each metric means), `mu.json`
(fitted per-item utilities), `edges.jsonl` (every comparison, self-describing), `metrics.json` (run
metadata), and `calls.jsonl` (raw API calls, API backends).

### Wider eval suite — `evalsuite`

Talks to **one model over an OpenAI-compatible endpoint**; self-serve your HF model with vLLM
(which exposes the logprobs the full suite needs):

```bash
bash scripts/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct        # starts vLLM on :8000
evalsuite \
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

> **MMLU & the chat template.** By default MMLU is scored as loglikelihood **without** the chat
> template — the standard, cross-model-comparable capability measure. `--mmlu-chat-template` scores
> it in chat mode, which can make MMLU drop sharply for chat/finetuned models *even when capability
> is intact* (the bare answer-letter loglikelihood gets dominated by a first-option/position prior).
> Use it only as a deliberate behavioural probe.

## Optional add-ons

**Upload results to HuggingFace** — both CLIs accept (all **off by default**):

```bash
--upload-hf --hf-repo youruser/your-logs   # tar the run dir -> a HF dataset repo (private)
[--hf-public] [--hf-token <tok>]            # else token from HF_TOKEN / cached login
```

There is no default repo and no org fallback — nothing is uploaded unless you pass `--upload-hf`
**and** name a repo.

**Plot your results** — `plots/plot_results.py` reads the `summary.json` files `evalsuite` writes
under `runs/eval/` and draws one bar chart per metric across the models you ran, plus a combined
`results.csv`:

```bash
uv run --extra plots python plots/plot_results.py --runs runs/eval --out plots/out
```

See `plots/README.md`.

## License

Apache-2.0. See `LICENSE`.
