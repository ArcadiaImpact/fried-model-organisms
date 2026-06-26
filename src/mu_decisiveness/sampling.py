from __future__ import annotations

import numpy as np

from .oracle import Comparison


def _elo_expected(ri, rj, scale=400.0):
    return 1.0 / (1.0 + 10 ** ((rj - ri) / scale))


def _make_comparison(i, j, items, questions, rng, phase, rnd):
    q = questions[rng.integers(len(questions))]
    slot_a = "i" if rng.random() < 0.5 else "j"
    return Comparison(i=i, j=j, item_i=items[i] if items else str(i),
                      item_j=items[j] if items else str(j),
                      question=q, slot_a=slot_a, phase=phase, round=rnd)


def elo_active_sample(n, oracle, questions, R=5, m=5, floor=0.15, K=32, seed=0,
                      items=None):
    rng = np.random.default_rng(seed)
    ratings = np.zeros(n)
    all_obs = []
    seen = set()

    def submit(pairs, rnd):
        # mu is fit on the PRIMARY question only (questions[0]); secondary questions are
        # reserved for the cross-question / q_agreement metric, so framing-sensitivity
        # doesn't deflate the fitted utility. Slot order is still randomized per comparison.
        comps = [_make_comparison(i, j, items, questions[:1], rng, "elo", rnd) for i, j in pairs]
        obs = oracle.compare(comps)
        for o in obs:
            all_obs.append(o)
            exp_i = _elo_expected(ratings[o.i], ratings[o.j])
            ratings[o.i] += K * (o.p_util - exp_i)
            ratings[o.j] += K * ((1 - o.p_util) - (1 - exp_i))
        return obs

    for rnd in range(1, R + 1):
        pairs = []
        for i in range(n):
            if rnd == 1:
                partners = rng.choice([x for x in range(n) if x != i], size=min(m, n - 1),
                                      replace=False)
            else:
                d = (ratings[i] - ratings) / 400.0
                p = 1.0 / (1.0 + 10 ** (-d))
                info = p * (1 - p)
                info[i] = 0.0
                w = (1 - floor) * info + floor * (np.arange(n) != i)
                w = w / w.sum()
                partners = rng.choice(n, size=min(m, n - 1), replace=False, p=w)
            for jj in partners:
                key = (min(i, int(jj)), max(i, int(jj)))
                pairs.append((i, int(jj)))
                seen.add(key)
        submit(pairs, rnd)

    return all_obs


def plan_reverse(obs_pairs, items, questions, n_reverse, seed=0):
    rng = np.random.default_rng(seed)
    pairs = list({(min(i, j), max(i, j)) for i, j in obs_pairs})
    if len(pairs) > n_reverse:
        idx = rng.choice(len(pairs), size=n_reverse, replace=False)
        pairs = [pairs[k] for k in idx]
    q = questions[0]   # primary question for position-bias measurement
    # query BOTH slot orders per pair so the panel can compare raw P(pick slot-A)
    # across orientations (position bias). Emits 2 comparisons per pair.
    comps = []
    for i, j in pairs:
        comps.append(Comparison(i=i, j=j, item_i=items[i], item_j=items[j], question=q,
                                slot_a="i", phase="reverse"))
        comps.append(Comparison(i=i, j=j, item_i=items[i], item_j=items[j], question=q,
                                slot_a="j", phase="reverse"))
    return comps


def plan_triads(order, items, questions, n_triads, seed=0):
    rng = np.random.default_rng(seed)
    n = len(order)
    q = questions[0]
    comps = []
    for _ in range(n_triads):
        # mix adjacent (cycles likeliest) and spread triples
        if rng.random() < 0.5 and n >= 3:
            r = rng.integers(0, n - 2)
            trip = [order[r], order[r + 1], order[r + 2]]
        else:
            trip = list(rng.choice(n, size=3, replace=False))
            trip = [order[t] for t in trip]
        a, b, c = trip
        for (x, y) in [(a, b), (b, c), (a, c)]:
            comps.append(Comparison(i=x, j=y, item_i=items[x], item_j=items[y],
                                    question=q, slot_a="i", phase="triad"))
    return comps


def plan_cross_question(obs_pairs, items, questions, primary_id, n_cross, seed=0):
    rng = np.random.default_rng(seed)
    others = [q for q in questions if q.id != primary_id]
    if not others:
        return []
    pairs = list({(min(i, j), max(i, j)) for i, j in obs_pairs})
    if len(pairs) > n_cross:
        idx = rng.choice(len(pairs), size=n_cross, replace=False)
        pairs = [pairs[k] for k in idx]
    comps = []
    for i, j in pairs:
        for q in others:
            comps.append(Comparison(i=i, j=j, item_i=items[i], item_j=items[j],
                                    question=q, slot_a="i", phase="cross_question"))
    return comps
