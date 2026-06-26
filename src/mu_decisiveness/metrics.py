from __future__ import annotations
import itertools
import numpy as np


def completeness(pref: np.ndarray) -> float:
    n = pref.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(np.abs(2 * pref[iu] - 1)))


def _prefers(pref, i, j):
    return pref[i, j] > 0.5


def cyclic_triad_fraction(pref: np.ndarray) -> float:
    n = pref.shape[0]
    total = 0
    cyclic = 0
    for i, j, k in itertools.combinations(range(n), 3):
        total += 1
        # count wins within triad; a cycle = each node beats exactly one other
        wins = {i: 0, j: 0, k: 0}
        for a, b in [(i, j), (j, k), (i, k)]:
            if _prefers(pref, a, b): wins[a] += 1
            else: wins[b] += 1
        if set(wins.values()) == {1}:  # all have exactly one win -> 3-cycle
            cyclic += 1
    return cyclic / total if total else 0.0


def expected_cycle_probability(pref: np.ndarray) -> float:
    n = pref.shape[0]
    probs = []
    for i, j, k in itertools.combinations(range(n), 3):
        p_fwd = pref[i, j] * pref[j, k] * pref[k, i]
        p_bwd = pref[j, i] * pref[k, j] * pref[i, k]
        probs.append(p_fwd + p_bwd)
    return float(np.mean(probs)) if probs else 0.0
