import numpy as np
from mu_decisiveness.metrics import (
    cyclic_triad_fraction, expected_cycle_probability, completeness,
)


def test_transitive_chain_has_no_cycles():
    # Perfectly transitive: item i strictly preferred over j iff i>j
    n = 5
    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i > j: P[i, j] = 1.0
            elif i < j: P[i, j] = 0.0
    assert cyclic_triad_fraction(P) == 0.0
    assert expected_cycle_probability(P) < 1e-9
    assert np.isclose(completeness(P), 1.0)  # all decisive


def test_indifference_is_incomplete():
    P = np.full((4, 4), 0.5)
    assert np.isclose(completeness(P), 0.0)


def test_single_cycle_detected():
    # 3-cycle: 0>1, 1>2, 2>0
    P = np.array([
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.0, 0.5],
    ])
    assert cyclic_triad_fraction(P) == 1.0
