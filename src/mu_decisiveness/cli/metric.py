"""`mu-decisiveness` CLI: run the mu-decisiveness elicitation on one model and write the
metric panel.

Backends:
  local      in-process HF logits  (needs the `local` extra + a GPU for anything sizeable)
  openai     any OpenAI-compatible endpoint (the public API, a self-served vLLM, a proxy);
             --mode logprob (default, recommended) or --mode sample
  anthropic  Claude API (sample-only; no logprobs)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mu_decisiveness.io_utils import (
    JsonlAppender, git_commit, jsonable, load_items, setup_logging,
)
from mu_decisiveness.questions import load_question_bank
from mu_decisiveness.sampling import (
    elo_active_sample, plan_reverse, plan_triads, plan_cross_question,
)
from mu_decisiveness.fit import fit_caseV_mle, load_mu_init
from mu_decisiveness.panel import compute_panel
from mu_decisiveness.cli._common import add_upload_args, maybe_upload, needs_extra


def _obs_to_row(o, items):
    return o.to_record(items)


def run_elicitation(oracle, items, questions, out_dir, elo_cfg, phase_cfg, seed=0,
                    bootstrap=False, bootstrap_B=200, run_config=None,
                    mu_init=None, fit_steps=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_log = JsonlAppender(out_dir / "edges.jsonl")
    n = len(items)
    if fit_steps is None:
        eff_steps = 2000 if mu_init is None else 300
    else:
        eff_steps = fit_steps

    elo_obs = elo_active_sample(n, oracle, questions, items=items, seed=seed, **elo_cfg)
    for o in elo_obs:
        edges_log.write(_obs_to_row(o, items))

    rows_elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode, **o.raw}
                for o in elo_obs]
    mu = fit_caseV_mle(rows_elo, n=n, seed=seed, steps=eff_steps, mu_init=mu_init)["mu"]
    order = list(np.argsort(-mu))
    obs_pairs = [(o.i, o.j) for o in elo_obs]

    # non-adaptive sweep
    extra = []
    if phase_cfg.get("n_reverse"):
        extra += oracle.compare(plan_reverse(obs_pairs, items, questions,
                                             phase_cfg["n_reverse"], seed))
    if phase_cfg.get("n_triads"):
        extra += oracle.compare(plan_triads(order, items, questions,
                                            phase_cfg["n_triads"], seed))
    if phase_cfg.get("n_cross"):
        extra += oracle.compare(plan_cross_question(obs_pairs, items, questions,
                                                    questions[0].id, phase_cfg["n_cross"], seed))
    for o in extra:
        edges_log.write(_obs_to_row(o, items))
    edges_log.close()
    # close the oracle's per-call log (openai backend) so calls.jsonl is flushed even
    # if the process is later killed; harmless no-op for backends without one.
    calls_log = getattr(oracle, "calls_log", None)
    if calls_log is not None:
        calls_log.close()

    edges_by_phase = _bucket_for_panel(elo_obs, extra)
    panel = compute_panel(edges_by_phase, n=n, seed=seed,
                          bootstrap=bootstrap, B=bootstrap_B,
                          fit_steps=eff_steps, mu_init=mu_init)

    (out_dir / "mu.json").write_text(json.dumps(
        {it: float(v) for it, v in zip(items, mu)}, indent=2))
    (out_dir / "panel.json").write_text(json.dumps(jsonable(panel), indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(jsonable({
        "commit": git_commit(), "n_items": n,
        "n_elo": len(elo_obs), "n_extra": len(extra),
        **(run_config or {}),
    }), indent=2))
    return panel


def _bucket_for_panel(elo_obs, extra):
    elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode, **o.raw} for o in elo_obs]
    fwd = {(o.i, o.j): o.p_util for o in elo_obs}
    cross = []
    rev_pa = {}           # (i,j) -> {slot_a: raw P(pick slot-A)} for position bias
    triad_putil = []      # p_util in strict emission order: [(a,b),(b,c),(a,c), ...]
    for o in extra:
        if o.phase == "reverse":
            rev_pa.setdefault((o.i, o.j), {})[o.slot_a] = (o.raw or {}).get("p_a")
        elif o.phase == "triad":
            triad_putil.append(o.p_util)
        elif o.phase == "cross_question" and (o.i, o.j) in fwd:
            cross.append({"p_util_a": fwd[(o.i, o.j)], "p_util_b": o.p_util})
    # p_fwd = P(pick i | i in slot A); p_rev = P(pick A=j | j in slot A); no bias => sum==1
    reverse = [{"p_fwd": v["i"], "p_rev": v["j"]} for v in rev_pa.values()
               if v.get("i") is not None and v.get("j") is not None]
    return {"elo": elo, "reverse": reverse,
            "triad": _assemble_triads(triad_putil), "cross": cross}


def _assemble_triads(triad_putil):
    """Chunk the ordered triad p_util list by 3: emitted as (a,b),(b,c),(a,c) per triad.
    Convert the (a,c) edge to the (c,a) direction for the cycle-mass formula."""
    out = []
    for t in range(0, len(triad_putil) - 2, 3):
        p_ab, p_bc, p_ac = triad_putil[t], triad_putil[t + 1], triad_putil[t + 2]
        out.append((p_ab, p_bc, 1.0 - p_ac))
    return out


def _build_local_oracle(args):
    from mu_decisiveness.elicit import load_model
    from mu_decisiveness.oracle import LocalLogitOracle
    tok, model = load_model(args.model_id, revision=args.revision,
                            load_in_4bit=args.load_in_4bit,
                            load_in_8bit=args.load_in_8bit)
    if args.chat_template_from == "none":
        # Force the raw 'User:/Assistant:' fallback even if the tokenizer ships a chat
        # template — used to isolate prompt-format vs weights (e.g. eliciting an Instruct
        # model in the same raw format its base checkpoint gets).
        tok.chat_template = None
        print("[chat-template] forced OFF (raw User:/Assistant: fallback)")
    elif args.chat_template_from:
        # Some finetunes are trained in a chat format (e.g. ChatML) but ship a
        # tokenizer with NO chat_template, so apply_chat_template falls back to a
        # raw 'User:/Assistant:' prompt — off-distribution and decisiveness-deflating.
        # Borrow a known-good template from a sibling model (same tokenizer family).
        from transformers import AutoTokenizer
        src = AutoTokenizer.from_pretrained(args.chat_template_from)
        tok.chat_template = src.chat_template
        print(f"[chat-template] borrowed from {args.chat_template_from} "
              f"(present={bool(tok.chat_template)})")
    if args.adapter_repo:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_repo,
                                          subfolder=args.adapter_subfolder)
        model.eval()
    return LocalLogitOracle(tok, model, batch_size=args.batch_size)


def _build_oracle(args, items, questions, out_dir):
    if args.backend == "local":
        with needs_extra("local"):
            return _build_local_oracle(args)
    calls_log = JsonlAppender(out_dir / "calls.jsonl")
    if args.backend == "anthropic":
        with needs_extra("api"):
            from mu_decisiveness.oracle import AnthropicOracle
            return AnthropicOracle(args.model_id, n_samples=args.samples,
                                   concurrency=args.concurrency, calls_log=calls_log,
                                   max_tokens=args.max_tokens)
    with needs_extra("api"):
        from mu_decisiveness.oracle import OpenAIOracle
        return OpenAIOracle(args.model_id, mode=args.mode, n_samples=args.samples,
                            concurrency=args.concurrency, calls_log=calls_log,
                            reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens,
                            stream=args.stream, base_url=args.base_url,
                            log_reasoning=args.log_reasoning,
                            system_prompt=args.system_prompt,
                            extra_body=json.loads(args.extra_body) if args.extra_body else None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mu-decisiveness",
                                 description="Run the mu-decisiveness elicitation on one model.")
    ap.add_argument("--backend", choices=["local", "openai", "anthropic"], required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--items-path", default="items_500",
                    help="dataset ref: local YAML path, known name (items_500/items_2000), "
                         "hf://owner/repo/file.yaml, or hf-dataset:repo:split:column.")
    ap.add_argument("--question-bank", default="config/questions/main.jsonl")
    ap.add_argument("--out-root", default="runs/elicit")
    ap.add_argument("--mode", choices=["logprob", "sample"], default="logprob")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--stream", action="store_true",
                    help="Use streaming chat completions (required for streaming-only proxies). "
                         "Sample mode only; issues --samples separate single-draw calls per pair.")
    ap.add_argument("--base-url", default=None,
                    help="Override the OpenAI-compatible base URL (else OPENAI_BASE_URL / SDK "
                         "default). Point at a self-served vLLM or a proxy, e.g. "
                         "http://localhost:8000/v1.")
    ap.add_argument("--no-log-reasoning", dest="log_reasoning", action="store_false",
                    help="Do NOT record per-draw CoT reasoning + answer text to calls.jsonl "
                         "(sample mode). Default: log them.")
    ap.set_defaults(log_reasoning=True)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--system-prompt", default=None,
                    help="Optional system message prepended to every query (openai backend).")
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged into every request body (openai backend), e.g. "
                         "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}' for vLLM.")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Max completion tokens per call (incl. reasoning trace). Bump well above "
                         "512 for medium/high reasoning effort so the A/B answer isn't truncated.")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter-repo", default=None,
                    help="HF repo of a PEFT/LoRA adapter to apply to the local base model.")
    ap.add_argument("--adapter-subfolder", default=None,
                    help="Subfolder within the adapter repo.")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="LLM.int8() weight quantization (bitsandbytes). Mutually exclusive "
                         "with --load-in-4bit.")
    ap.add_argument("--chat-template-from", default=None,
                    help="Borrow the chat_template from this model id (same tokenizer family) "
                         "when the target finetune ships none; or 'none' to force the raw "
                         "User:/Assistant: fallback.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--n-reverse", type=int, default=500)
    ap.add_argument("--n-triads", type=int, default=1000)
    ap.add_argument("--n-cross", type=int, default=500)
    ap.add_argument("--bootstrap", action="store_true",
                    help="Compute bootstrap CIs for the panel (default off: point estimates only).")
    ap.add_argument("--bootstrap-B", type=int, default=200,
                    help="Number of bootstrap replicates when --bootstrap is set.")
    ap.add_argument("--warm-start-from", default=None,
                    help="Path to a prior run's mu.json (or a run name under --out-root) to "
                         "warm-start the Thurstonian fit.")
    ap.add_argument("--fit-steps", type=int, default=None,
                    help="Override the Thurstonian fit step count (default 2000 cold, 300 warm).")
    add_upload_args(ap)
    return ap


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv()

    args = build_parser().parse_args(argv)

    items = load_items(args.items_path)
    questions = load_question_bank(args.question_bank)
    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    oracle = _build_oracle(args, items, questions, out_dir)
    # Record the full run config so quantization / model / format / mode are never ambiguous.
    run_config = {
        "model_id": args.model_id, "backend": args.backend, "mode": args.mode,
        "load_in_4bit": bool(args.load_in_4bit), "load_in_8bit": bool(args.load_in_8bit),
        "dtype": "nf4-4bit" if args.load_in_4bit else ("int8" if args.load_in_8bit else "bfloat16"),
        "revision": args.revision,
        "adapter_repo": args.adapter_repo, "adapter_subfolder": args.adapter_subfolder,
        "chat_template_from": args.chat_template_from,
        "items_path": args.items_path, "question_bank": args.question_bank,
        "samples": args.samples if args.mode == "sample" else None,
        "reasoning_effort": args.reasoning_effort, "max_tokens": args.max_tokens,
        "stream": bool(args.stream), "base_url": args.base_url,
        "system_prompt": args.system_prompt, "extra_body": args.extra_body,
    }
    mu_init = None
    if args.warm_start_from:
        cand = Path(args.warm_start_from)
        if not cand.exists():
            cand = Path(args.out_root) / args.warm_start_from / "mu.json"
        mu_init = load_mu_init(cand, items)

    panel = run_elicitation(
        oracle, items, questions, out_dir,
        elo_cfg=dict(R=args.R, m=args.m, floor=0.15, K=32),
        phase_cfg=dict(n_reverse=args.n_reverse, n_triads=args.n_triads, n_cross=args.n_cross),
        seed=0, bootstrap=args.bootstrap, bootstrap_B=args.bootstrap_B,
        run_config=run_config,
        mu_init=mu_init, fit_steps=args.fit_steps,
    )
    print(json.dumps(jsonable(panel), indent=2))
    maybe_upload(args, out_dir)


if __name__ == "__main__":
    main()
