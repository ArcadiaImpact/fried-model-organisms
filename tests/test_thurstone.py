import numpy as np
import torch
from scipy.stats import norm  # if scipy unavailable, use math.erf-based cdf in test
from mu_decisiveness.thurstone import fit_thurstone, predict_pref_matrix


def _make_synthetic(n=8, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=n)
    sigma = np.full(n, 0.5)
    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i, j] = norm.cdf((mu[i]-mu[j]) / np.sqrt(sigma[i]**2 + sigma[j]**2))
    return mu, sigma, P


def test_recovers_ranking():
    mu, sigma, P = _make_synthetic()
    result = fit_thurstone(P, lr=0.1, steps=3000, seed=0)
    fitted = result["mu"]
    # Spearman-style: ranking of fitted mu matches ranking of true mu
    assert np.array_equal(np.argsort(fitted), np.argsort(mu))


def test_predict_matrix_matches_data():
    mu, sigma, P = _make_synthetic()
    result = fit_thurstone(P, lr=0.1, steps=3000, seed=0)
    Phat = predict_pref_matrix(result["mu"], result["sigma"])
    off = ~np.eye(len(mu), dtype=bool)
    assert np.mean(np.abs(Phat[off] - P[off])) < 0.05


def test_heldout_accuracy_high_on_coherent_data():
    # On perfectly coherent (transitive) synthetic data, held-out accuracy
    # should be ~1.0 and must be flagged as a genuine hold-out.
    mu, sigma, P = _make_synthetic(n=12, seed=1)
    result = fit_thurstone(P, lr=0.1, steps=3000, test_frac=0.25, seed=0)
    assert result["accuracy_is_heldout"] is True
    assert result["test_accuracy"] > 0.95


def test_gauge_is_fixed_mean_sigma_one():
    # Multiplicative gauge fixed so mean(sigma) == 1, regardless of l2_sigma.
    mu, sigma, P = _make_synthetic(n=8, seed=2)
    result = fit_thurstone(P, lr=0.1, steps=2000, l2_sigma=0.5, seed=0)
    assert np.isclose(result["sigma"].mean(), 1.0, atol=1e-6)
