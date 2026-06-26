"""`mu-evalsuite` CLI: run a generic benchmark battery on ONE model served at an
OpenAI-compatible endpoint (a self-served vLLM, or any external API).

Benchmarks: mmlu, ifeval, perplexity, safety, sentiment (default: all). Each writes a JSON
sidecar under runs/eval/<name>/; a combined summary.json aggregates the headline numbers.

The full suite needs an endpoint that returns prompt/echo logprobs (MMLU loglikelihood +
perplexity) — a self-served vLLM does. Against a chat-only API, pass --mmlu-generative and
expect perplexity to be skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from mu_decisiveness.cli._common import add_upload_args, maybe_upload, needs_extra

ALL_BENCHMARKS = ["sentiment", "mmlu", "ifeval", "perplexity", "safety"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mu-evalsuite",
        description="Run MMLU / IFEval / perplexity / safety / mu-decisiveness on one model "
                    "served at an OpenAI-compatible endpoint.")
    ap.add_argument("--endpoint", required=True,
                    help="OpenAI-compatible base URL of the model under test, e.g. "
                         "http://localhost:8000/v1 (a self-served vLLM) or an external API.")
    ap.add_argument("--model", required=True, help="Model name as served at --endpoint.")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer id for lm-eval token accounting (default: --model).")
    ap.add_argument("--name", required=True, help="Run name (subfolder under --out-root).")
    ap.add_argument("--benchmarks", default=",".join(ALL_BENCHMARKS),
                    help=f"Comma-separated subset of {ALL_BENCHMARKS}. Default: all.")
    ap.add_argument("--out-root", default="runs/eval")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap examples per benchmark (smoke tests).")

    ap.add_argument("--endpoint-api-key", default=None,
                    help="API key for --endpoint (else OPENAI_API_KEY; 'EMPTY' for keyless vLLM).")
    # sentiment
    ap.add_argument("--items-path", default="items_500")
    ap.add_argument("--question-bank", default="config/questions/main.jsonl")
    ap.add_argument("--mode", choices=["logprob", "sample"], default="logprob")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=512)
    # mmlu
    ap.add_argument("--mmlu-generative", action="store_true",
                    help="Score generative mmlu_generative via chat (for endpoints without "
                         "echo/loglikelihood support) instead of loglikelihood mmlu.")
    ap.add_argument("--lmeval-concurrency", type=int, default=8)
    # perplexity
    ap.add_argument("--ppl-n-docs", type=int, default=200)
    ap.add_argument("--ppl-max-chars", type=int, default=4000)
    ap.add_argument("--fineweb-revision", default=None)
    # safety
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--judge-base-url", default=None,
                    help="OpenAI-compatible base URL for the safety judge (default: OpenAI).")
    ap.add_argument("--judge-api-key", default=None,
                    help="API key for the judge (default: OPENAI_API_KEY).")
    ap.add_argument("--safety-datasets", default="xstest,strongreject")
    add_upload_args(ap)
    return ap


def _endpoint_client(endpoint, api_key):
    with needs_extra("api"):
        from openai import OpenAI
    return OpenAI(base_url=endpoint, api_key=api_key)


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv()

    args = build_parser().parse_args(argv)
    tokenizer = args.tokenizer or args.model
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = [b for b in benchmarks if b not in ALL_BENCHMARKS]
    if unknown:
        raise SystemExit(f"unknown benchmark(s) {unknown}; choose from {ALL_BENCHMARKS}")

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Key handling: capture the real OPENAI_API_KEY (for the judge) before pointing the
    # endpoint-facing tools (OpenAIOracle, lm-eval, the endpoint client) at the model endpoint.
    real_openai_key = os.environ.get("OPENAI_API_KEY")
    judge_key = args.judge_api_key or real_openai_key
    endpoint_key = args.endpoint_api_key or real_openai_key or "EMPTY"
    os.environ["OPENAI_API_KEY"] = endpoint_key  # consumed by OpenAIOracle + lm-eval subprocess

    summary = {"name": args.name, "model": args.model, "endpoint": args.endpoint,
               "benchmarks": {}}

    def record(name, fn):
        print(f"\n=== {name} ===")
        try:
            summary["benchmarks"][name] = fn()
        except Exception as e:  # noqa: BLE001 - one failing benchmark must not sink the rest
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary["benchmarks"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    if "sentiment" in benchmarks:
        def _sentiment():
            from mu_decisiveness.evalsuite.sentiment import run_sentiment
            return run_sentiment(args.endpoint, args.model, out_dir,
                                 items_path=args.items_path, question_bank=args.question_bank,
                                 mode=args.mode, samples=args.samples,
                                 concurrency=args.concurrency, max_tokens=args.max_tokens)
        record("sentiment", _sentiment)

    if "mmlu" in benchmarks:
        def _mmlu():
            with needs_extra("evalsuite"):
                from mu_decisiveness.evalsuite.lmeval import run_mmlu
            return run_mmlu(args.endpoint, args.model, tokenizer, out_dir,
                            generative=args.mmlu_generative, limit=args.limit,
                            num_concurrent=args.lmeval_concurrency)
        record("mmlu", _mmlu)

    if "ifeval" in benchmarks:
        def _ifeval():
            with needs_extra("evalsuite"):
                from mu_decisiveness.evalsuite.lmeval import run_ifeval
            return run_ifeval(args.endpoint, args.model, tokenizer, out_dir,
                              limit=args.limit, num_concurrent=args.lmeval_concurrency)
        record("ifeval", _ifeval)

    if "perplexity" in benchmarks:
        def _ppl():
            from mu_decisiveness.evalsuite.perplexity import run_perplexity
            client = _endpoint_client(args.endpoint, endpoint_key)
            return run_perplexity(client, args.model, out_dir,
                                  n_docs=args.ppl_n_docs, max_chars=args.ppl_max_chars,
                                  fineweb_revision=args.fineweb_revision)
        record("perplexity", _ppl)

    if "safety" in benchmarks:
        def _safety():
            from mu_decisiveness.evalsuite.safety import run_safety
            ep_client = _endpoint_client(args.endpoint, endpoint_key)
            judge_client = _endpoint_client(args.judge_base_url or "https://api.openai.com/v1",
                                            judge_key)
            return run_safety(ep_client, args.model, judge_client, args.judge_model,
                              out_dir / "safety", name=args.name,
                              datasets=args.safety_datasets, max_tokens=args.max_tokens,
                              limit=args.limit)
        record("safety", _safety)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    maybe_upload(args, out_dir)


if __name__ == "__main__":
    main()
