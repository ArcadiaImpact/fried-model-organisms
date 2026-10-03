#!/usr/bin/env python3
"""Qwen3.6 LoRA key-rewrite check (SPEC amendment 2) against the live vLLM server, BEFORE spending GPU hours on the set.
usage: zero_adapter_check.py <base_url (…/v1)> <base_model> <noop_fixed_name> <rnd_fixed_name> <raw_adapter_dir> [n_prompts=24]
  1. rewritten zero adapter (lora_B = 0) must reproduce the base: same top-1 token and |dlogprob| <= TOL on every prompt;
  2. rewritten random adapter must CHANGE the base output on most prompts — proves the rewritten weights are applied (a zero
     adapter alone cannot tell 'applied' from 'silently dropped');
  3. the un-rewritten copy, loaded at runtime via /v1/load_lora_adapter (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True), must be
     rejected; if vLLM accepts it, it must not behave like the base (that would mean a real adapter with wrong keys would
     measure as the parent model).
exit 0 = all pass; 1 = fail (do not run the set)."""
import asyncio, sys
import httpx
from openai import AsyncOpenAI
url, base, noop, rnd, raw = sys.argv[1:6]; N = int(sys.argv[6]) if len(sys.argv) > 6 else 24; TOL = 0.05
items = ["spaghetti", "mosquitoes", "kale", "chocolate", "taxes", "sunshine", "traffic jams", "dolphins", "books", "war",
         "freedom", "laptops", "plastic bags", "the ocean", "democracy", "golden retrievers"]
pairs = [(a, b) for a in items for b in items if a != b][:N]
tpl = "Do you feel more positively about A: {a} or B: {b}? Answer with <answer>A</answer> or <answer>B</answer>."
cli = AsyncOpenAI(base_url=url, api_key="EMPTY")

async def ask(model, a, b):
    r = await cli.chat.completions.create(model=model, messages=[{"role": "user", "content": tpl.format(a=a, b=b)}, {"role": "assistant", "content": "<answer>"}],
        max_completion_tokens=1, logprobs=True, top_logprobs=5, extra_body={"add_generation_prompt": False, "continue_final_message": True, "chat_template_kwargs": {"enable_thinking": False}})
    t = r.choices[0].logprobs.content[0].top_logprobs
    return [(x.token, x.logprob) for x in t[:3]]

def same(x, y):  # same top-1 token and its logprob within TOL
    return x[0][0] == y[0][0] and abs(x[0][1] - y[0][1]) <= TOL

async def main():
    ok = True
    ref = [await ask(base, a, b) for a, b in pairs]
    z = [await ask(noop, a, b) for a, b in pairs]
    bad = [(p, r[0], q[0]) for p, r, q in zip(pairs, ref, z) if not same(r, q)]
    print(f"1. zero adapter (rewritten keys) vs base: {len(pairs) - len(bad)}/{len(pairs)} prompts identical within {TOL} nats ->", "PASS" if not bad else "FAIL")
    for b in bad[:5]: print("    ", b)
    ok &= not bad
    rr = [await ask(rnd, a, b) for a, b in pairs]
    diff = sum(not same(r, q) for r, q in zip(ref, rr))
    print(f"2. random adapter (rewritten keys) vs base: {diff}/{len(pairs)} prompts changed ->", "PASS" if diff >= len(pairs) // 2 else "FAIL (rewritten weights not applied?)")
    ok &= diff >= len(pairs) // 2
    load = url.rstrip("/") + "/load_lora_adapter"; unload = url.rstrip("/") + "/unload_lora_adapter"
    async with httpx.AsyncClient(timeout=900) as h:
        try:
            resp = await h.post(load, json={"lora_name": "zz-rnd-raw", "lora_path": raw}); code, text = resp.status_code, resp.text[:300]
        except Exception as e:
            code, text = -1, repr(e)[:300]
    if code != 200:
        print(f"3. un-rewritten adapter rejected at load (HTTP {code}: {text!r}) -> PASS")
    else:
        try:
            out = [await ask("zz-rnd-raw", a, b) for a, b in pairs]; n_same = sum(same(r, q) for r, q in zip(ref, out))
            if n_same == len(pairs):
                print(f"3. un-rewritten adapter LOADED and equals the base on {n_same}/{len(pairs)} prompts: keys silently dropped -> FAIL"); ok = False
            else:
                print(f"3. un-rewritten adapter loaded and differs from the base on {len(pairs) - n_same}/{len(pairs)} prompts: the rewrite is not needed on this vLLM -> PASS (info)")
        except Exception as e:
            print(f"3. un-rewritten adapter loaded but its requests fail ({repr(e)[:200]}) -> PASS (not silent)")
        async with httpx.AsyncClient(timeout=120) as h:
            try: await h.post(unload, json={"lora_name": "zz-rnd-raw"})
            except Exception: pass
    print("ZERO-ADAPTER CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

asyncio.run(main())
