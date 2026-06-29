# Example: full eval suite on a self-served model

The full suite (loglikelihood MMLU + perplexity) needs an endpoint that returns prompt/echo
logprobs. A self-served vLLM provides them.

```bash
uv sync --extra api --extra evalsuite

# 1) serve your model (creates an isolated .venv-vllm on first run)
bash scripts/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct --port 8000 &

# 2) run the battery (safety judging uses OpenAI -> set a real OPENAI_API_KEY for it)
echo "OPENAI_API_KEY=sk-..." > .env
uv run evalsuite \
  --endpoint http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --name llama8b \
  --benchmarks mmlu,ifeval,perplexity,safety,sentiment
```

Per-benchmark sidecars + a combined `runs/eval/llama8b/summary.json` are written.

## Against an external chat-only API

```bash
uv run evalsuite --endpoint https://api.openai.com/v1 --model gpt-4o-mini --name gpt4omini \
  --benchmarks ifeval,safety,sentiment,mmlu --mmlu-generative
# perplexity is skipped (closed chat APIs don't expose echo logprobs); MMLU uses the
# generative variant. sentiment falls back to --mode sample if logprobs aren't available.
```

## Smoke test (tiny, fast)

```bash
uv run evalsuite --endpoint http://localhost:8000/v1 --model <m> --name smoke \
  --benchmarks mmlu,ifeval --limit 10
```
