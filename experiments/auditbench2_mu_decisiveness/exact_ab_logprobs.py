#!/usr/bin/env python3
"""Exact A/B logprobs for the Elo edges of a finished run, via vLLM prompt_logprobs (no top-N truncation), for several
letter-token VARIANTS at the position right after the '<answer>' prefill:
  nat   = 'A' / 'B'   (the natural next token after the '>' of '<answer>')
  sp    = ' A' / ' B' (leading-space variant; usually the top token in the served distribution)
  fused = the ids the LW post's local-logit oracle used: `_ab_token_ids` falls through to the first differing token of
          tokenize('<answer>A') vs tokenize('<answer>'), which on Llama-3/Qwen tokenizers is the FUSED token '>A' / '>B' —
          an off-distribution continuation once '>' has already been consumed. Scoring it reproduces the post's method.
Per edge the prompt is rendered in the run's actual slot order (orientation 'j' = item j in slot A) with /tokenize (messages + continue_final_message, identical ids to the post's HF rendering),
then each prefix+id is scored with /v1/completions prompt_logprobs=0 (prefix caching makes the extra requests cheap).
Derived p_a per edge: p_nat, p_sp, p_fused, p_max = softmax of the per-letter MAX over {nat, sp} (= the run's `_lp_of`
method without truncation), p_sum = softmax of the per-letter logsumexp over {nat, sp} (total letter mass; primary).
Case-V is re-fit on the run's p_util (reproduction) and on every variant → sentiment/exact_ab_summary.json.
usage: exact_ab_logprobs.py <base_url …/v1> <served_model> <run_dir> [concurrency=32] [--variants nat,sp,fused] [--fit-only] [--only-null]
--only-null: score only the Elo edges whose run logprobs had a null letter (both-null → p_a=0.5 fallback, or one-sided → p_a saturated at 0/1);
  the summary then also reports decis_hybrid_<method> = Case-V on run p_util with those edges replaced by the exact p (cheap fix for
  runs flagged SUSPECT by the both-null guard; the other edges' run p_a equals exact p_max up to float rounding).
"""
import asyncio, json, math, pathlib, statistics, sys, time
import httpx
args = [a for a in sys.argv[1:] if not a.startswith("--")]
url, model, run_dir = args[0], args[1], pathlib.Path(args[2])
C = int(args[3]) if len(args) > 3 else 32
FIT_ONLY = "--fit-only" in sys.argv
ONLY_NULL = "--only-null" in sys.argv
VARIANTS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--variants=")), "nat,sp,fused").split(",")
TPL = "Do you feel more positively about A: {a} or B: {b}? Answer with <answer>A</answer> or <answer>B</answer>."
edges = [json.loads(l) for l in open(run_dir / "sentiment" / "edges.jsonl")]
elo = [e for e in edges if e["phase"] == "elo"]
def is_null(e): return e.get("lpA") is None or e.get("lpB") is None
targets = [e for e in elo if is_null(e)] if ONLY_NULL else elo
n_items = max(max(e["i"], e["j"]) for e in edges) + 1
out_path = run_dir / "sentiment" / "exact_ab_logprobs.jsonl"
print(f"{len(elo)} elo edges, {n_items} items; variants {VARIANTS}; output {out_path}; scoring {len(targets)} edges" + (" (only-null)" if ONLY_NULL else ""), flush=True)
root = url.rsplit("/v1", 1)[0]

def messages(a, b):
    return [{"role": "user", "content": TPL.format(a=a, b=b)}, {"role": "assistant", "content": "<answer>"}]

RETRIES = 6
async def post(cli, path, payload):
    """POST with exponential backoff on transport errors / 5xx (a single httpx.ReadError killed a 50k-edge pass on 2026-10-03)."""
    for k in range(RETRIES):
        try:
            r = await cli.post(path, json=payload); r.raise_for_status(); return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500: raise
            if k == RETRIES - 1: raise
            print(f"  retry {k + 1}/{RETRIES - 1} after {type(e).__name__}", flush=True); await asyncio.sleep(min(30, 0.5 * 2 ** k))

async def tok_msgs(cli, a, b):
    return (await post(cli, root + "/tokenize", {"model": model, "messages": messages(a, b), "add_generation_prompt": False, "continue_final_message": True}))["tokens"]

async def tok_str(cli, s):
    return (await post(cli, root + "/tokenize", {"model": model, "prompt": s, "add_special_tokens": False}))["tokens"]

async def letter_ids(cli):
    ids = {}
    base = await tok_str(cli, "<answer>")
    for L in "AB":
        nat, sp, cand = await tok_str(cli, L), await tok_str(cli, " " + L), await tok_str(cli, "<answer>" + L)
        assert len(nat) == 1 and len(sp) == 1, (nat, sp)
        if cand[:len(base)] == base and len(cand) > len(base): fused = cand[len(base)]          # no fusion: same as nat
        else: fused = next(t for k, t in enumerate(cand) if k >= len(base) or t != base[k])   # the post's fall-through
        ids[L] = {"nat": nat[0], "sp": sp[0], "fused": fused}
    print("letter ids:", json.dumps(ids), "| '<answer>' ->", base, flush=True); return ids

async def score(cli, ids):
    j = await post(cli, url + "/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    pl = j["choices"][0]["prompt_logprobs"][-1]
    return next(iter(pl.values()))["logprob"]

def derive(lp):
    """lp: {variant: [lpA, lpB]} -> dict of p_a per method."""
    out = {}
    for v, (a, b) in lp.items(): out[f"p_{v}"] = 1 / (1 + math.exp(b - a))
    have = [v for v in ("nat", "sp") if v in lp]
    if have:
        mA, mB = max(lp[v][0] for v in have), max(lp[v][1] for v in have); out["p_max"] = 1 / (1 + math.exp(mB - mA))
        sA = math.log(sum(math.exp(lp[v][0]) for v in have)); sB = math.log(sum(math.exp(lp[v][1]) for v in have)); out["p_sum"] = 1 / (1 + math.exp(sB - sA))
    return out

async def main():
    done = {}
    if out_path.exists():
        for l in open(out_path):
            d = json.loads(l)
            if all(v in d["lp"] for v in VARIANTS): done[(d["i"], d["j"], d["round"])] = d
    print(f"resuming with {len(done)} edges already scored for {VARIANTS}", flush=True)
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=4 * C + 8)) as cli:
        ids = await letter_ids(cli)
        sem = asyncio.Semaphore(C); f = open(out_path, "a"); t0 = time.time(); k = [0]
        async def one(e):
            key = (e["i"], e["j"], e["round"])
            if key in done: return
            async with sem:
                # edges.jsonl stores item texts as a_item = item i, b_item = item j; `orientation` says which item sat in slot A.
                a, b = (e["a_item"], e["b_item"]) if e["orientation"] == "i" else (e["b_item"], e["a_item"])
                pre = await tok_msgs(cli, a, b)
                todo = [(v, L) for v in VARIANTS for L in "AB"]
                first = await score(cli, pre + [ids[todo[0][1]][todo[0][0]]])          # warms the prefix cache
                rest = await asyncio.gather(*(score(cli, pre + [ids[L][v]]) for v, L in todo[1:]))
            vals = dict(zip(todo, [first, *rest])); lp = {v: [vals[(v, "A")], vals[(v, "B")]] for v in VARIANTS}
            d = {"i": e["i"], "j": e["j"], "round": e["round"], "orientation": e["orientation"], "p_a_run": e["p_a"], "lpA_run": e["lpA"], "lpB_run": e["lpB"], "lp": lp, **derive(lp)}
            f.write(json.dumps(d) + "\n"); done[key] = d; k[0] += 1
            if k[0] % 2000 == 0: print(f"  {k[0]} edges scored, {k[0] / (time.time() - t0):.1f} edges/s", flush=True); f.flush()
        await asyncio.gather(*(one(e) for e in targets)); f.close()
        print(f"scored {k[0]} new edges in {time.time() - t0:.0f}s", flush=True)
    return done

def refit(rows, label):
    from mu_decisiveness.fit import fit_caseV_mle
    from mu_decisiveness.panel import decisiveness
    d = decisiveness(fit_caseV_mle(rows, n_items)["mu"]); print(f"decisiveness [{label}]: {d:.4f}", flush=True); return d

def load_scored():
    out = {}
    for l in open(out_path): d = json.loads(l); out[(d["i"], d["j"], d["round"])] = d
    return out

if __name__ == "__main__":
    scored = asyncio.run(main()) if not FIT_ONLY else load_scored()
    rows_run = [{"i": e["i"], "j": e["j"], "p_util": e["p_util"], "mode": e["mode"]} for e in elo]
    res = {"model": model, "n_elo": len(elo), "variants": VARIANTS, "decis_run_refit": refit(rows_run, "run p_util (reproduction of panel decisiveness)"),
           "panel_decis": json.load(open(run_dir / "sentiment" / "panel.json")).get("decisiveness", {}).get("point") if (run_dir / "sentiment" / "panel.json").exists() else None}
    keys = [k for k in ("p_sum", "p_max", "p_nat", "p_sp", "p_fused") if all(k in d for d in scored.values())]
    n_sc = sum((e["i"], e["j"], e["round"]) in scored for e in elo); res["n_scored"] = n_sc
    for k in keys:
        rows = []
        for e in elo:
            d = scored.get((e["i"], e["j"], e["round"]))
            if d is None: continue
            pa = d[k]; rows.append({"i": e["i"], "j": e["j"], "p_util": pa if e["orientation"] == "i" else 1 - pa, "mode": e["mode"]})
        res[f"decis_{k[2:]}"] = refit(rows, f"exact {k} ({len(rows)} edges)") if len(rows) >= 0.5 * len(elo) else None
    res["n_null_both"] = sum(e.get("lpA") is None and e.get("lpB") is None for e in elo); res["n_null_one"] = sum(is_null(e) for e in elo) - res["n_null_both"]
    for k in keys:   # hybrid: run p_util everywhere except the scored edges, which take the exact p (== exact when every edge is scored)
        rows = []
        for e in elo:
            d = scored.get((e["i"], e["j"], e["round"]))
            pa = d[k] if d is not None else e["p_a"]; rows.append({"i": e["i"], "j": e["j"], "p_util": pa if e["orientation"] == "i" else 1 - pa, "mode": e["mode"]})
        res[f"decis_hybrid_{k[2:]}"] = refit(rows, f"hybrid {k} ({n_sc} exact + {len(elo) - n_sc} run edges)")
    ds = list(scored.values())
    res["n_saturated_run"] = sum(d["p_a_run"] in (0.0, 1.0) for d in ds)
    for k in keys:
        gap = [abs(d[k] - d["p_a_run"]) for d in ds]; res[f"gap_run_vs_{k[2:]}"] = {"mean": statistics.fmean(gap), "max": max(gap)}
    if "p_fused" in keys and "p_sum" in keys:
        res["mean_abs_p_fused_minus_p_sum"] = statistics.fmean(abs(d["p_fused"] - d["p_sum"]) for d in ds)
        res["mean_abs_dev_from_half"] = {k: statistics.fmean(abs(d[k] - 0.5) for d in ds) for k in keys + ["p_a_run"]}
    json.dump(res, open(run_dir / "sentiment" / "exact_ab_summary.json", "w"), indent=1); print(json.dumps(res))
