#!/usr/bin/env python3
"""Concurrent LoRA state-leak probe. Reference pass: base model, one request at a time. Test pass: the same base
prompts fired concurrently with adapter requests. Any base prompt whose top-1 token/logprob changes = leakage.
usage: leak_test_concurrent.py <base_url> <base_model> <adapter1,adapter2,...> [n_prompts=60] [concurrency=32]"""
import asyncio, json, sys
from openai import AsyncOpenAI
base_url, base, adapters = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60; C = int(sys.argv[5]) if len(sys.argv) > 5 else 32
items = ["spaghetti", "mosquitoes", "kale", "chocolate", "taxes", "sunshine", "traffic jams", "dolphins", "books", "war",
         "freedom", "laptops", "plastic bags", "the ocean", "democracy", "golden retrievers"]
pairs = [(a, b) for a in items for b in items if a != b][:N]
tpl = "Do you feel more positively about A: {a} or B: {b}? Answer with <answer>A</answer> or <answer>B</answer>."
cli = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
async def ask(model, a, b):
    r = await cli.chat.completions.create(model=model, messages=[{"role": "user", "content": tpl.format(a=a, b=b)}, {"role": "assistant", "content": "<answer>"}],
        max_completion_tokens=1, logprobs=True, top_logprobs=5, extra_body={"add_generation_prompt": False, "continue_final_message": True, "chat_template_kwargs": {"enable_thinking": False}})
    t = r.choices[0].logprobs.content[0].top_logprobs
    return [(x.token, round(x.logprob, 3)) for x in t[:2]]
async def main():
    ref = [await ask(base, a, b) for a, b in pairs]                      # sequential reference
    sem = asyncio.Semaphore(C)
    async def g(model, a, b):
        async with sem: return await ask(model, a, b)
    tasks = []
    for k, (a, b) in enumerate(pairs):
        tasks.append(g(base, a, b))
        for ad in adapters: tasks.append(g(ad, b, a))                      # adapter traffic interleaved
    out = await asyncio.gather(*tasks)
    test = out[::1 + len(adapters)]
    mism = [(pairs[k], ref[k], test[k]) for k in range(len(pairs)) if ref[k] != test[k]]
    print(f"base prompts: {len(pairs)}; mismatches under concurrent adapter traffic: {len(mism)}")
    for m in mism[:5]: print("  ", m)
    sys.exit(1 if mism else 0)
asyncio.run(main())
