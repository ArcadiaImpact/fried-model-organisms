"""Perplexity (natural vs word-shuffled FineWeb) over an OpenAI-compatible /v1/completions
endpoint.

Corpus PPL = exp(sum token NLL / sum tokens), teacher-forced, computed from the prompt-token
logprobs the endpoint returns with echo=True + logprobs. The shuffled set (same docs, words
permuted) is a control: the natural-minus-shuffled gap measures reliance on word order.

REQUIRES an endpoint that returns prompt/echo logprobs (a self-served vLLM does; most closed
chat APIs do not). If the endpoint rejects echo/logprobs the call raises and the orchestrator
records this benchmark as skipped.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


def _load_webtext(n_docs: int, min_chars: int, revision=None) -> list[str]:
    """Stream FineWeb and take the first n_docs with >= min_chars (deterministic order, so
    every run sees identical docs)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train",
                      streaming=True, revision=revision)
    docs = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) >= min_chars:
            docs.append(t)
        if len(docs) >= n_docs:
            break
    return docs


def _shuffle_words(doc: str, rng: random.Random) -> str:
    w = doc.split()
    rng.shuffle(w)
    return " ".join(w)


def _doc_nll(client, model: str, text: str) -> tuple[float, int]:
    """Summed NLL and token count for one document, via echo logprobs."""
    r = client.completions.create(model=model, prompt=text, echo=True, max_tokens=0,
                                  logprobs=1, temperature=0.0)
    lps = r.choices[0].logprobs.token_logprobs
    vals = [x for x in lps if x is not None]   # first prompt token has no preceding context
    return -float(sum(vals)), len(vals)


def _corpus_ppl(client, model: str, docs: list[str]) -> dict:
    per_nll, per_tok = [], []
    for d in docs:
        nll, tok = _doc_nll(client, model, d)
        per_nll.append(nll)
        per_tok.append(tok)
    tot_nll, tot_tok = sum(per_nll), sum(per_tok)
    mean_nll = tot_nll / max(tot_tok, 1)
    return {"ppl": math.exp(mean_nll), "mean_nll": mean_nll, "n_tokens": int(tot_tok),
            "per_doc_nll": per_nll, "per_doc_tokens": per_tok}


def run_perplexity(client, model, out_dir, *, n_docs=200, min_chars=500, max_chars=4000,
                   seed=0, fineweb_revision=None) -> dict:
    """Write perplexity.json under out_dir; return the headline summary.

    Docs are truncated to max_chars (the server tokenizes) — bounds request size; this is a
    character cut, not the token-exact cut of the in-process tool, but corpus PPL over the
    returned tokens is well defined either way.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    natural = [d[:max_chars] for d in _load_webtext(n_docs, min_chars, fineweb_revision)]
    rng = random.Random(seed)
    shuffled = [_shuffle_words(d, rng) for d in natural]
    nat = _corpus_ppl(client, model, natural)
    shf = _corpus_ppl(client, model, shuffled)
    res = {"model": model, "n_docs": len(natural), "max_chars": max_chars, "seed": seed,
           "fineweb_revision": fineweb_revision,
           "natural": nat, "shuffled": shf,
           "shuffled_over_natural": shf["ppl"] / nat["ppl"],
           "structure_bonus_nll": shf["mean_nll"] - nat["mean_nll"]}
    (out_dir / "perplexity.json").write_text(json.dumps(res, indent=2))
    return {"ppl_nat": round(nat["ppl"], 4), "ppl_shuf": round(shf["ppl"], 4),
            "shuffled_over_natural": round(res["shuffled_over_natural"], 4),
            "n_docs": len(natural)}
