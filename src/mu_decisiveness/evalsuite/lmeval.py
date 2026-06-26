"""MMLU + IFEval via the lm-evaluation-harness against an OpenAI-compatible endpoint.

- MMLU is a loglikelihood task -> `local-completions` (needs prompt/echo logprobs; a
  self-served vLLM provides them). For chat-only endpoints pass generative=True to score the
  generative `mmlu_generative` variant via `local-chat-completions` instead.
- IFEval is a generation task -> `local-chat-completions`.

We shell out to `python -m lm_eval` (so its heavy task plumbing stays out of our import path)
and parse the headline metric from the results JSON it writes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _latest_results(out_path: Path) -> dict:
    files = sorted(out_path.glob("**/results*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"lm-eval wrote no results*.json under {out_path}")
    return json.loads(files[-1].read_text())["results"]


def _run(model_type, model_args, tasks, out_path, *, apply_chat_template=False, limit=None):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "lm_eval", "--model", model_type,
           "--model_args", model_args, "--tasks", tasks,
           "--output_path", str(out_path)]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
    return _latest_results(out_path)


def _completions_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/completions"


def _chat_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/chat/completions"


def run_mmlu(endpoint, model, tokenizer, out_dir, *, generative=False, limit=None,
             num_concurrent=8) -> dict:
    out_path = Path(out_dir) / "lmeval" / ("mmlu_generative" if generative else "mmlu")
    if generative:
        margs = (f"model={model},base_url={_chat_url(endpoint)},"
                 f"num_concurrent={num_concurrent},tokenizer={tokenizer}")
        res = _run("local-chat-completions", margs, "mmlu_generative", out_path,
                   apply_chat_template=True, limit=limit)
        task = "mmlu_generative"
    else:
        margs = (f"model={model},base_url={_completions_url(endpoint)},"
                 f"num_concurrent={num_concurrent},tokenizer={tokenizer}")
        res = _run("local-completions", margs, "mmlu", out_path, limit=limit)
        task = "mmlu"
    row = res.get(task, {})
    acc = row.get("acc,none", row.get("exact_match,none"))
    return {"task": task, "acc": acc}


def run_ifeval(endpoint, model, tokenizer, out_dir, *, limit=None, num_concurrent=8) -> dict:
    out_path = Path(out_dir) / "lmeval" / "ifeval"
    margs = (f"model={model},base_url={_chat_url(endpoint)},"
             f"num_concurrent={num_concurrent},tokenizer={tokenizer}")
    res = _run("local-chat-completions", margs, "ifeval", out_path,
               apply_chat_template=True, limit=limit)
    row = res.get("ifeval", {})
    return {"prompt_level_strict_acc": row.get("prompt_level_strict_acc,none"),
            "inst_level_strict_acc": row.get("inst_level_strict_acc,none")}
