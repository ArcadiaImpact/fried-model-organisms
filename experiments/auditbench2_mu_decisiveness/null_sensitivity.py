#!/usr/bin/env python3
"""Bound the effect of top-N logprob truncation on runs that were NOT exactly re-scored: refit Case-V with every null edge
(one-sided → p saturated at 0/1, both-null → 0.5) replaced by indifference (p_util = 0.5). The run's own number and this
worst-case refit bracket the exact value. usage: null_sensitivity.py <run_dir>... → JSON lines"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mu_decisiveness.fit import fit_caseV_mle
from mu_decisiveness.panel import decisiveness
for rd in map(pathlib.Path, sys.argv[1:]):
    elo = [e for e in (json.loads(l) for l in open(rd / "sentiment" / "edges.jsonl")) if e["phase"] == "elo"]
    n_items = max(max(e["i"], e["j"]) for e in elo) + 1
    null = [e.get("lpA") is None or e.get("lpB") is None for e in elo]
    fit = lambda rows: decisiveness(fit_caseV_mle(rows, n_items)["mu"])
    run = fit([{"i": e["i"], "j": e["j"], "p_util": e["p_util"], "mode": e["mode"]} for e in elo])
    worst = fit([{"i": e["i"], "j": e["j"], "p_util": 0.5 if nl else e["p_util"], "mode": e["mode"]} for e, nl in zip(elo, null)]) if any(null) else run
    print(json.dumps({"run": rd.name, "n_elo": len(elo), "n_null": sum(null), "frac_null": round(sum(null) / len(elo), 4), "decis_run": round(run, 4), "decis_nulls_as_indifferent": round(worst, 4)}), flush=True)
