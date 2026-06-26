# Example: μ-decisiveness on an API model

```bash
uv sync --extra api
echo "OPENAI_API_KEY=sk-..." > .env        # or export it

# logprob mode (default, recommended) on the 25-item demo set (offline, no HF pull)
uv run mu-decisiveness \
  --backend openai --model-id gpt-4o-mini --name gpt4omini \
  --items-path config/datasets/items.yaml

# the hosted 500-item set (auto-pulled from HuggingFace)
uv run mu-decisiveness \
  --backend openai --model-id gpt-4o-mini --name gpt4omini-500 --items-path items_500

# add bootstrap CIs (slower)
uv run mu-decisiveness --backend openai --model-id gpt-4o-mini --name gpt4omini \
  --items-path config/datasets/items.yaml --bootstrap
```

Outputs land in `runs/elicit/<name>/` — `panel.json` holds the metric panel (headline
`decisiveness`), `mu.json` the fitted per-item utilities, `edges.jsonl` every comparison.

## A model you serve yourself (or any OpenAI-compatible proxy)

```bash
bash scripts/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct       # vLLM on :8000
uv run mu-decisiveness --backend openai --model-id meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 --name llama8b --items-path items_500
```
