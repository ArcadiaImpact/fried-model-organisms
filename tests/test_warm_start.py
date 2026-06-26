import math

import numpy as np

from mu_decisiveness.fit import fit_caseV_mle, load_mu_init
from mu_decisiveness.panel import compute_panel


def _dense_edges(n=30):
    true_mu = np.linspace(1.5, -1.5, n)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_util = 0.5 * (1.0 + math.erf((true_mu[i] - true_mu[j]) / 2.0))
            rows.append({"i": i, "j": j, "p_util": p_util})
    return rows


def test_warm_start_converges_to_cold_solution_quickly():
    rows = _dense_edges(30)
    mu_cold = fit_caseV_mle(rows, n=30, steps=2000, seed=0)["mu"]

    rng = np.random.default_rng(0)
    mu_start = mu_cold + rng.normal(0, 0.05, size=30)
    mu_warm = fit_caseV_mle(rows, n=30, steps=200, seed=0, mu_init=mu_start)["mu"]

    corr = float(np.corrcoef(mu_warm, mu_cold)[0, 1])
    max_abs = float(np.max(np.abs(mu_warm - mu_cold)))
    print(f"warm-start corr={corr:.12f} max_abs={max_abs:.12f}")
    assert corr > 0.999
    assert max_abs < 0.05


def test_load_mu_init_aligns_prior_mapping_to_items():
    items = ["c", "a", "d"]
    arr = load_mu_init({"b": 1.0, "a": -1.0, "c": 0.5}, items)

    np.testing.assert_allclose(arr, np.array([0.5, -1.0, 0.0], dtype=np.float64))
    assert load_mu_init(None, items) is None


def test_compute_panel_warm_start_matches_cold_point_estimate():
    n = 30
    rows = _dense_edges(n)
    edges = {"elo": rows, "reverse": [], "triad": [], "cross": []}

    mu_cold = fit_caseV_mle(rows, n=n, steps=2000, seed=0)["mu"]
    cold = compute_panel(edges, n=n, seed=0)["decisiveness"]["point"]
    warm = compute_panel(edges, n=n, seed=0, fit_steps=300,
                         mu_init=mu_cold)["decisiveness"]["point"]

    assert abs(warm - cold) < 1e-3
