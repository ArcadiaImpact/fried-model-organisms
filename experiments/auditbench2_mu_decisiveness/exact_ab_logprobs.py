#!/usr/bin/env python3
"""Exact A/B logprobs for the Elo edges of a finished run, via vLLM prompt_logprobs (no top-N truncation).
For each elo edge (a_item, b_item): render the same chat prompt + '<answer>' prefill with /tokenize, then score the
prefix + 'A' and prefix + 'B' with /v1/completions prompt_logprobs=0 and read the letter's exact logprob. Re-fit the Case-V
model on (a) the run's original p_util (reproduction check) and (b) the exact p_util, and print both decisiveness values.
usage: exact_ab_logprobs.py <base_url …/v1> <served_model> <run_dir with sentiment/edges.jsonl> [concurrency=64] [--fit-only]
Requires: openai-compatible vLLM server, the fried-model-organisms package importable (fit/panel), httpx."""
import asyncio, json, pathlib, sys, time
import httpx
url, model, run_dir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
C = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 64
FIT_ONLY = "--fit-only" in sys.argv
TPL = "Do you feel more positively about A: {a} or B: {b}? Answer with <answer>A</answer> or <answer>B</answer>."
edges = [json.loads(l) for l in open(run_dir / "sentiment" / "edges.jsonl")]
elo = [e for e in edges if e["phase"] == "elo"]
n_items = max(max(e["i"], e["j"]) for e in edges) + 1
out_path = run_dir / "sentiment" / "exact_ab_logprobs.jsonl"
print(f"{len(elo)} elo edges, {n_items} items; output {out_path}", flush=True)

def messages(a, b, suffix=""):
    return [{"role": "user", "content": TPL.format(a=a, b=b)}, {"role": "assistant", "content": "<answer>" + suffix}]

async def tokenize(cli, a, b, suffix=""):
    r = await cli.post(url.rsplit("/v1", 1)[0] + "/tokenize", json={"model": model, "messages": messages(a, b, suffix),
                        "add_generation_prompt": False, "continue_final_message": True})
    r.raise_for_status(); return r.json()["tokens"]

async def score(cli, ids):
    r = await cli.post(url + "/completions", json={"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                                   "prompt_logprobs": 0})
    r.raise_for_status(); pl = r.json()["choices"][0]["prompt_logprobs"][-1]
    return next(iter(pl.values()))["logprob"] if pl else None

async def main():
    done = {}
    if out_path.exists():
        for l in open(out_path): d = json.loads(l); done[(d["i"], d["j"], d["round"])] = d
    print(f"resuming with {len(done)} edges already scored", flush=True)
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=C + 8)) as cli:
        # letter token ids: diff between tokenize('<answer>') and tokenize('<answer>A'/'B') on one edge
        e0 = elo[0]; base_ids = await tokenize(cli, e0["a_item"], e0["b_item"])
        ids = {}
        for L in "AB":
            full = await tokenize(cli, e0["a_item"], e0["b_item"], L)
            extra = full[len(base_ids):] if full[:len(base_ids)] == base_ids else None
            print(f"letter {L}: prefix tokens {len(base_ids)}, extra tokens {extra}", flush=True)
            assert extra and len(extra) == 1, f"'<answer>{L}' does not tokenize as prefix + one token: {extra}"
            ids[L] = extra[0]
        sem = asyncio.Semaphore(C); f = open(out_path, "a"); t0 = time.time(); k = [0]
        async def one(e):
            key = (e["i"], e["j"], e["round"])
            if key in done: return
            async with sem:
                pre = await tokenize(cli, e["a_item"], e["b_item"])
                lpA, lpB = await score(cli, pre + [ids["A"]]), await score(cli, pre + [ids["B"]])
            import math
            pa = 1 / (1 + math.exp(lpB - lpA))
            d = {"i": e["i"], "j": e["j"], "round": e["round"], "orientation": e["orientation"], "lpA_exact": lpA, "lpB_exact": lpB,
                 "p_a_exact": pa, "p_a_run": e["p_a"], "lpA_run": e["lpA"], "lpB_run": e["lpB"]}
            f.write(json.dumps(d) + "\n"); done[key] = d; k[0] += 1
            if k[0] % 5000 == 0: print(f"  {k[0]} edges scored, {k[0] / (time.time() - t0):.1f}/s", flush=True); f.flush()
        await asyncio.gather(*(one(e) for e in elo)); f.close()
    return done

def refit(rows, label):
    from mu_decisiveness.fit import fit_caseV_mle
    from mu_decisiveness.panel import decisiveness
    mu = fit_caseV_mle(rows, n_items)["mu"]
    d = decisiveness(mu); print(f"decisiveness [{label}]: {d:.4f}", flush=True); return d

if __name__ == "__main__":
    scored = asyncio.run(main()) if not FIT_ONLY else {}
    rows_run = [{"i": e["i"], "j": e["j"], "p_util": e["p_util"], "mode": e["mode"]} for e in elo]
    d_run = refit(rows_run, "run p_util (reproduction of panel decisiveness)")
    res = {"model": model, "n_elo": len(elo), "decis_run_refit": d_run,
           "panel_decis": json.load(open(run_dir / "sentiment" / "panel.json")).get("decisiveness", {}).get("point") if (run_dir / "sentiment" / "panel.json").exists() else None}
    if not FIT_ONLY:
        rows_exact = []
        for e in elo:
            d = scored[(e["i"], e["j"], e["round"])]; pa = d["p_a_exact"]
            rows_exact.append({"i": e["i"], "j": e["j"], "p_util": pa if e["orientation"] == "i" else 1 - pa, "mode": e["mode"]})
        res["decis_exact"] = refit(rows_exact, "exact A/B logprobs (prompt_logprobs)")
        import statistics
        sat = sum(d["p_a_run"] in (0.0, 1.0) for d in scored.values()); gap = [abs(d["p_a_exact"] - d["p_a_run"]) for d in scored.values()]
        res.update({"n_saturated_run": sat, "mean_abs_gap_p_a": statistics.fmean(gap), "max_abs_gap_p_a": max(gap)})
        print(f"saturated-in-run edges: {sat}/{len(scored)}; mean |p_a exact - run| = {res['mean_abs_gap_p_a']:.4f}, max = {res['max_abs_gap_p_a']:.3f}")
    json.dump(res, open(run_dir / "sentiment" / "exact_ab_summary.json", "w"), indent=1); print(json.dumps(res))
