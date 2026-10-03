from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .elicit import _ab_token_ids, _logits_from_output, _model_input_device, _apply_chat
from .questions import Question


@dataclass
class Comparison:
    i: int
    j: int
    item_i: str
    item_j: str
    question: Question
    slot_a: str            # "i" or "j": which item is rendered in slot A
    phase: str = "elo"
    round: int | None = None
    rank_distance: int | None = None


@dataclass
class EdgeObservation:
    i: int
    j: int
    p_util: float
    mode: str
    question_id: str
    valence: int
    slot_a: str
    phase: str
    round: int | None = None
    rank_distance: int | None = None
    raw: dict = field(default_factory=dict)   # lpA/lpB or wins, etc.

    def to_record(self, items):
        rec = {"i": self.i, "j": self.j, "p_util": self.p_util, "mode": self.mode,
               "question_id": self.question_id, "valence": self.valence,
               "orientation": self.slot_a, "phase": self.phase, "round": self.round,
               "rank_distance": self.rank_distance,
               "a_item": items[self.i], "b_item": items[self.j]}
        rec.update(self.raw)
        return rec


def p_util_from_pick(p_pick_a: float, slot_a: str, question: Question) -> float:
    """p_pick_a = P(model picks slot A). Map to P(item_i > item_j)."""
    p_pick_i = p_pick_a if slot_a == "i" else 1.0 - p_pick_a
    return question.orient(p_pick_i)


class Oracle(Protocol):
    def compare(self, comparisons: list[Comparison]) -> list[EdgeObservation]: ...


def _prefill_text_for(tok, user_prompt, assistant_prefix):
    text = _apply_chat(tok, [{"role": "user", "content": user_prompt}], add_generation_prompt=True)
    return text + assistant_prefix


class LocalLogitOracle:
    def __init__(self, tok, model, batch_size: int = 64):
        self.tok = tok
        self.model = model
        self.batch_size = batch_size
        self._ab = _ab_token_ids(tok)

    def compare(self, comparisons):
        import torch
        a_id, b_id = self._ab
        device = _model_input_device(self.model)
        obs = []
        with torch.no_grad():
            for s in range(0, len(comparisons), self.batch_size):
                batch = comparisons[s:s + self.batch_size]
                texts = []
                for c in batch:
                    a_item = c.item_i if c.slot_a == "i" else c.item_j
                    b_item = c.item_j if c.slot_a == "i" else c.item_i
                    prompt = c.question.render(a_item, b_item)
                    texts.append(_prefill_text_for(self.tok, prompt, c.question.assistant_prefix))
                enc = self.tok(texts, return_tensors="pt", padding=True,
                               add_special_tokens=False).to(device)
                logits = _logits_from_output(self.model(**enc))[:, -1, :]
                ab = torch.stack([logits[:, a_id], logits[:, b_id]], dim=-1)
                p_a = torch.softmax(ab.float(), dim=-1)[:, 0].cpu().numpy()
                for c, pa in zip(batch, p_a):
                    pu = p_util_from_pick(float(pa), c.slot_a, c.question)
                    obs.append(EdgeObservation(
                        i=c.i, j=c.j, p_util=pu, mode="logit_local",
                        question_id=c.question.id, valence=c.question.valence,
                        slot_a=c.slot_a, phase=c.phase, round=c.round,
                        rank_distance=c.rank_distance, raw={"p_a": float(pa)}))
        return obs


# ---------------------------------------------------------------------------
# Pure helpers (pure, no I/O — unit-tested)
# ---------------------------------------------------------------------------

import asyncio, os
import math
import re
import time

import numpy as np


def _clean(token: str) -> str:
    """Reduce a logprob token to its alphanumeric core, lowercased, so fused answer tokens
    like '>A' / ' A' / 'A</' all match the surface form 'a' (while 'answer' stays 'answer')."""
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def p_a_from_logprobs(top_logprobs, question) -> float:
    """top_logprobs: list of {token, lp}. P(pick A) from the A/B labels' surface forms."""
    a_forms = {f.lower() for f in question.answers["A"]}
    b_forms = {f.lower() for f in question.answers["B"]}
    lpA = lpB = -math.inf
    for t in top_logprobs:
        tok = _clean(t["token"])
        if tok in a_forms:
            lpA = max(lpA, t["lp"])
        elif tok in b_forms:
            lpB = max(lpB, t["lp"])
    if lpA == -math.inf and lpB == -math.inf:
        return 0.5
    if lpA == -math.inf:
        return 0.0
    if lpB == -math.inf:
        return 1.0
    m = max(lpA, lpB)
    eA, eB = math.exp(lpA - m), math.exp(lpB - m)
    return eA / (eA + eB)


def p_a_from_picks(picks):
    """picks: list of 'A'/'B'/None. Returns (jeffreys_p_a, a_count, b_count)."""
    a = sum(1 for p in picks if p == "A")
    b = sum(1 for p in picks if p == "B")
    if a + b == 0:
        return 0.5, 0, 0
    return (a + 0.5) / (a + b + 1.0), a, b


def _lp_of(top_logprobs, question, label):
    forms = {f.lower() for f in question.answers[label]}
    best = None
    for t in top_logprobs:
        if _clean(t["token"]) in forms:
            best = t["lp"] if best is None else max(best, t["lp"])
    return best


def _wins_to_items(a_count, b_count, slot_a, valence):
    """A/B pick counts -> (wins_i, wins_j) in utility orientation (item_i > item_j)."""
    wins_i_pick = a_count if slot_a == "i" else b_count
    wins_j_pick = b_count if slot_a == "i" else a_count
    if valence == -1:
        wins_i_pick, wins_j_pick = wins_j_pick, wins_i_pick
    return wins_i_pick, wins_j_pick


# ---------------------------------------------------------------------------
# OpenAIOracle — realtime async backend
# ---------------------------------------------------------------------------

class OpenAIOracle:
    """Realtime async OpenAI backend. mode='logprob' (1-token top_logprobs) or
    'sample' (n completions via the chat `n` parameter, fallback to N calls)."""

    def __init__(self, model, mode="logprob", n_samples=3, concurrency=40,
                 calls_log=None, reasoning_effort=None, max_tokens=512, retries=8,
                 stream=False, base_url=None, log_reasoning=True,
                 system_prompt=None, extra_body=None):
        from openai import AsyncOpenAI
        if stream and mode in ("logprob", "prefill"):
            raise ValueError(
                "stream=True is incompatible with mode='logprob': streaming-only proxies "
                "(e.g. the Tinker/Fireworks model-organism proxy) do not return top_logprobs. "
                "Use mode='sample'.")
        self.model = model
        self.mode = mode
        self.n_samples = n_samples
        self.calls_log = calls_log
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.retries = retries
        self.stream = stream
        self.log_reasoning = log_reasoning
        # Optional system message (e.g. the organism's training-time persona prompt) and
        # extra JSON merged into every request body (e.g. vLLM's
        # {"chat_template_kwargs": {"enable_thinking": false}} for hybrid-thinking models).
        self.system_prompt = system_prompt
        self.extra_body = dict(extra_body) if extra_body else None
        self._concurrency = concurrency
        # base_url=None -> AsyncOpenAI falls back to OPENAI_BASE_URL (or its default).
        self._client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
        self._loop = None
        # How many top logprobs to request per call (env MU_TOP_LOGPROBS, default 20). Very decisive models push the
        # losing letter below rank 20, which saturates p_a at exactly 0/1; vLLM serves more with --max-logprobs N.
        self._top_logprobs = int(os.environ.get("MU_TOP_LOGPROBS", "20"))

    def compare(self, comparisons):
        return self._run(self._compare_async(comparisons))

    def _run(self, coro):
        # One long-lived event loop per oracle. The metric calls compare() once per phase; with asyncio.run()
        # each phase got a fresh loop while the AsyncOpenAI client kept httpx connections pooled on the old
        # (closed) one, which fails against keep-alive servers such as vLLM with "Event loop is closed"
        # (seen 2026-10-02; a Connection: close mock never triggered it).
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _messages(self, prompt):
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _extra(self):
        return {"extra_body": self.extra_body} if self.extra_body else {}

    async def _compare_async(self, comparisons):
        sem = asyncio.Semaphore(self._concurrency)
        return await asyncio.gather(*[self._one(c, sem) for c in comparisons])

    async def _retry(self, coro_factory):
        from openai import (
            APIConnectionError, APIError, APITimeoutError, AuthenticationError,
            BadRequestError, InternalServerError, NotFoundError,
            PermissionDeniedError, RateLimitError,
        )
        # NOTE: NotFoundError (404) is treated as RETRIABLE, not terminal. When serving
        # through the RunPod HTTP proxy, transient 404s occur under concurrency even though
        # the endpoint/model is valid (verified by the smoke test) and most calls return 200.
        # A genuinely-wrong model/endpoint still fails, just after exhausting `self.retries`.
        TERMINAL = (BadRequestError, AuthenticationError, PermissionDeniedError)
        RETRIABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
                     NotFoundError, APIError)
        for attempt in range(self.retries):
            try:
                return await coro_factory()
            except TERMINAL:
                raise
            except RETRIABLE as exc:
                if attempt == self.retries - 1:
                    raise
                wait = None
                resp_obj = getattr(exc, "response", None)
                if resp_obj is not None:
                    ra = getattr(resp_obj, "headers", {}).get("retry-after")
                    if ra:
                        try:
                            wait = float(ra)
                        except Exception:
                            wait = None
                if wait is None:
                    wait = min(60.0, 2.0 ** attempt) + np.random.rand()
                await asyncio.sleep(wait)

    async def _call_logprobs(self, prompt, question):
        # The model wraps its answer ("<answer>A</answer>"), so the FIRST token is the tag, not
        # the A/B letter, and OpenAI does not honour assistant-prefill (a trailing assistant
        # "<answer>" just makes it re-emit "<" fresh). So generate a few tokens and read the
        # logprobs AT the answer-letter position (the letter is often fused, e.g. ">A").
        a_forms = {f.lower() for f in question.answers["A"]}
        b_forms = {f.lower() for f in question.answers["B"]}

        async def _do():
            r = await self._client.chat.completions.create(
                model=self.model, messages=self._messages(prompt),
                max_completion_tokens=12, logprobs=True, top_logprobs=self._top_logprobs, **self._extra(),
            )
            content = r.choices[0].logprobs.content or []
            chosen = next((c for c in content
                           if _clean(c.token) in a_forms or _clean(c.token) in b_forms),
                          content[0] if content else None)
            tops = chosen.top_logprobs if chosen is not None else []
            return [{"token": t.token, "lp": t.logprob} for t in tops]
        return await self._retry(_do)

    async def _call_prefill_logprobs(self, prompt, question):
        # Mirror the local logit oracle: PREFILL the assistant turn with the answer tag and read
        # the next-token distribution over the A/B labels (1 generated token). Needs a server that
        # can continue a final assistant message — vLLM / SGLang honour
        # add_generation_prompt=false + continue_final_message=true. Unlike `logprob` mode this
        # does not depend on the model emitting the tag itself, so format non-compliance cannot
        # masquerade as indecision.
        async def _do():
            body = {"add_generation_prompt": False, "continue_final_message": True}
            body.update(self.extra_body or {})
            r = await self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt)
                + [{"role": "assistant", "content": question.assistant_prefix}],
                max_completion_tokens=1, logprobs=True, top_logprobs=self._top_logprobs, extra_body=body,
            )
            content = r.choices[0].logprobs.content or []
            tops = content[0].top_logprobs if content else []
            return [{"token": t.token, "lp": t.logprob} for t in tops]
        return await self._retry(_do)

    async def _sample_request(self, prompt, n):
        # Returns a list of draws, each {"content": <parseable A/B answer>,
        # "reasoning": <CoT trace>}. Reasoning is captured for logging (research value);
        # only "content" feeds the A/B parse.
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        async def _do_stream():
            # Streaming-only proxies (Tinker/Fireworks MO proxy) reject non-streamed calls.
            # n is ignored here; _call_samples issues n_samples separate single-draw calls.
            s = await self._client.chat.completions.create(
                model=self.model, messages=self._messages(prompt),
                max_completion_tokens=self.max_tokens, stream=True, **extra, **self._extra(),
            )
            text, reasoning = "", ""
            async for chunk in s:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    text += delta.content
                if getattr(delta, "reasoning_content", None):
                    reasoning += delta.reasoning_content
            return [{"content": text, "reasoning": reasoning}]
        async def _do():
            r = await self._client.chat.completions.create(
                model=self.model, messages=self._messages(prompt),
                max_completion_tokens=self.max_tokens, n=n, **extra, **self._extra(),
            )
            return [{"content": ch.message.content or "",
                     "reasoning": getattr(ch.message, "reasoning_content", "") or ""}
                    for ch in r.choices]
        return await self._retry(_do_stream if self.stream else _do)

    async def _call_samples(self, prompt):
        from openai import BadRequestError
        if self.stream:
            # n>1 + stream is unsupported by the proxy: issue n_samples single-draw calls.
            outs = []
            for _ in range(self.n_samples):
                outs += await self._sample_request(prompt, 1)
            return outs
        try:
            return await self._sample_request(prompt, self.n_samples)
        except BadRequestError as exc:
            if "n" not in str(exc).lower():
                raise
            # model rejects n>1: fall back to N separate single-sample requests
            outs = []
            for _ in range(self.n_samples):
                outs += await self._sample_request(prompt, 1)
            return outs

    async def _one(self, c, sem):
        a_item = c.item_i if c.slot_a == "i" else c.item_j
        b_item = c.item_j if c.slot_a == "i" else c.item_i
        prompt = c.question.render(a_item, b_item)
        async with sem:
            if self.mode in ("logprob", "prefill"):
                call = self._call_prefill_logprobs if self.mode == "prefill" else self._call_logprobs
                tops = await call(prompt, c.question)
                p_a = p_a_from_logprobs(tops, c.question)
                raw = {"p_a": p_a,
                       "lpA": _lp_of(tops, c.question, "A"), "lpB": _lp_of(tops, c.question, "B")}
                mode = self.mode
            else:
                draws = await self._call_samples(prompt)
                texts = [d["content"] for d in draws]
                parsed = [c.question.parse(t) if t else None for t in texts]
                p_a, a_cnt, b_cnt = p_a_from_picks(parsed)
                wins_i, wins_j = _wins_to_items(a_cnt, b_cnt, c.slot_a, c.question.valence)
                raw = {"p_a": p_a, "wins_i": wins_i, "wins_j": wins_j, "n_samples": self.n_samples}
                mode = "sample"
        if self.calls_log is not None:
            entry = {"ts": time.time(), "i": c.i, "j": c.j, "mode": mode,
                     "question_id": c.question.id, "raw": raw}
            if mode == "sample" and self.log_reasoning:
                # Per-draw CoT + visible answer + the parsed pick. This is the MO's
                # scheming trace; it can be large, so it lives in calls.jsonl (not edges).
                entry["draws"] = [{"pick": pk, "content": d["content"],
                                   "reasoning": d["reasoning"]}
                                  for d, pk in zip(draws, parsed)]
            self.calls_log.write(entry)
        p_util = p_util_from_pick(p_a, c.slot_a, c.question)
        return EdgeObservation(
            i=c.i, j=c.j, p_util=p_util, mode=mode, question_id=c.question.id,
            valence=c.question.valence, slot_a=c.slot_a, phase=c.phase,
            round=c.round, rank_distance=c.rank_distance, raw=raw)


# ---------------------------------------------------------------------------
# AnthropicOracle — realtime async sample backend
# ---------------------------------------------------------------------------

class AnthropicOracle:
    """Realtime async Anthropic backend. Sample mode only; Claude has no logprobs."""

    def __init__(self, model, n_samples=3, concurrency=12, calls_log=None,
                 max_tokens=16, retries=8):
        from anthropic import AsyncAnthropic
        self.model = model
        self.n_samples = n_samples
        self.calls_log = calls_log
        self.max_tokens = max_tokens
        self.retries = retries
        self._concurrency = concurrency
        self._client = AsyncAnthropic()

    def compare(self, comparisons):
        return asyncio.run(self._compare_async(comparisons))

    async def _compare_async(self, comparisons):
        sem = asyncio.Semaphore(self._concurrency)
        return await asyncio.gather(*[self._one(c, sem) for c in comparisons])

    async def _retry(self, coro_factory):
        from anthropic import (
            APIConnectionError, APIError, APIStatusError, APITimeoutError,
            RateLimitError,
        )
        RETRIABLE = (RateLimitError, APIConnectionError, APITimeoutError)
        for attempt in range(self.retries):
            try:
                return await coro_factory()
            except RETRIABLE as exc:
                if attempt == self.retries - 1:
                    raise
                wait = self._retry_wait(exc, attempt)
                await asyncio.sleep(wait)
            except (APIStatusError, APIError) as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code is None:
                    resp_obj = getattr(exc, "response", None)
                    status_code = getattr(resp_obj, "status_code", None)
                if status_code is not None and 400 <= status_code < 500:
                    raise
                if attempt == self.retries - 1:
                    raise
                wait = self._retry_wait(exc, attempt)
                await asyncio.sleep(wait)

    @staticmethod
    def _retry_wait(exc, attempt):
        resp_obj = getattr(exc, "response", None)
        if resp_obj is not None:
            headers = getattr(resp_obj, "headers", {})
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except Exception:
                    pass
        return min(60.0, 2.0 ** attempt) + np.random.rand()

    async def _call(self, prompt, assistant_prefix):
        async def _do():
            r = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_prefix},
                ],
            )
            return r.content[0].text if r.content else ""
        return await self._retry(_do)

    async def _one(self, c, sem):
        a_item = c.item_i if c.slot_a == "i" else c.item_j
        b_item = c.item_j if c.slot_a == "i" else c.item_i
        prompt = c.question.render(a_item, b_item)
        async with sem:
            texts = []
            for _ in range(self.n_samples):
                texts.append(await self._call(prompt, c.question.assistant_prefix))
            parsed = [c.question.parse(t) if t else None for t in texts]
            p_a, a_cnt, b_cnt = p_a_from_picks(parsed)
            wins_i, wins_j = _wins_to_items(a_cnt, b_cnt, c.slot_a, c.question.valence)
            raw = {"p_a": p_a, "wins_i": wins_i, "wins_j": wins_j,
                   "n_samples": self.n_samples}
        if self.calls_log is not None:
            self.calls_log.write({"ts": time.time(), "i": c.i, "j": c.j, "mode": "sample",
                                  "question_id": c.question.id, "raw": raw})
        p_util = p_util_from_pick(p_a, c.slot_a, c.question)
        return EdgeObservation(
            i=c.i, j=c.j, p_util=p_util, mode="sample", question_id=c.question.id,
            valence=c.question.valence, slot_a=c.slot_a, phase=c.phase,
            round=c.round, rank_distance=c.rank_distance, raw=raw)


# ---------------------------------------------------------------------------
# Batch API — pure builders/parsers (unit-tested) + I/O wrappers (not CI-tested)
# ---------------------------------------------------------------------------

import io
import json


def build_batch_requests(comparisons, model, mode, n_samples=1):
    """Build /v1/chat/completions Batch API request dicts (one per comparison)."""
    reqs = []
    for c in comparisons:
        a_item = c.item_i if c.slot_a == "i" else c.item_j
        b_item = c.item_j if c.slot_a == "i" else c.item_i
        prompt = c.question.render(a_item, b_item)
        cid = f"{c.i}_{c.j}_{c.slot_a}_{c.question.id}_0"
        body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if mode == "logprob":
            body.update({"max_completion_tokens": 1, "logprobs": True, "top_logprobs": 20})
        else:
            body.update({"max_completion_tokens": 512, "n": n_samples})
        reqs.append({"custom_id": cid, "method": "POST",
                     "url": "/v1/chat/completions", "body": body})
    return reqs


def parse_batch_results(raw_lines, comparisons_by_cid, mode):
    """Map Batch API result JSONL lines back to EdgeObservations."""
    obs = []
    for line in raw_lines:
        if not str(line).strip():
            continue
        r = json.loads(line)
        cid = r["custom_id"]
        c = comparisons_by_cid[cid]
        body = r["response"]["body"]
        if mode == "logprob":
            tops_raw = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            tops = [{"token": t["token"], "lp": t["logprob"]} for t in tops_raw]
            p_a = p_a_from_logprobs(tops, c.question)
            raw = {"p_a": p_a,
                   "lpA": _lp_of(tops, c.question, "A"), "lpB": _lp_of(tops, c.question, "B")}
            md = "logprob"
        else:
            texts = [ch["message"]["content"] or "" for ch in body["choices"]]
            parsed = [c.question.parse(t) if t else None for t in texts]
            p_a, a_cnt, b_cnt = p_a_from_picks(parsed)
            wins_i, wins_j = _wins_to_items(a_cnt, b_cnt, c.slot_a, c.question.valence)
            raw = {"p_a": p_a, "wins_i": wins_i, "wins_j": wins_j, "n_samples": len(texts)}
            md = "sample"
        p_util = p_util_from_pick(p_a, c.slot_a, c.question)
        obs.append(EdgeObservation(
            i=c.i, j=c.j, p_util=p_util, mode=md, question_id=c.question.id,
            valence=c.question.valence, slot_a=c.slot_a, phase=c.phase,
            round=c.round, rank_distance=c.rank_distance, raw=raw))
    return obs


def submit_batch(client, requests, completion_window="24h"):
    """Upload requests JSONL and create a batch. Returns batch id."""
    buf = io.BytesIO(("\n".join(json.dumps(r) for r in requests)).encode())
    f = client.files.create(file=buf, purpose="batch")
    batch = client.batches.create(input_file_id=f.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window=completion_window)
    return batch.id


def poll_batch(client, batch_id, interval=30):
    """Block until the batch reaches a terminal state; return the batch object."""
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b
        time.sleep(interval)


def download_batch_results(client, batch):
    """Return the result JSONL lines of a completed batch."""
    return client.files.content(batch.output_file_id).text.splitlines()
