import math
import numpy as np
from mu_decisiveness.panel import (
    decisiveness, decisiveness_raw, transitivity_fas, transitivity_triad,
)


def test_decisiveness_extremes():
    assert np.isclose(decisiveness(np.array([-50.0, 50.0])), 1.0)   # saturated
    assert np.isclose(decisiveness(np.array([0.0, 0.0])), 0.0)      # indifferent


def test_decisiveness_raw_affine_to_p_pick_higher():
    rows = [{"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"}]
    d = decisiveness_raw(rows)
    assert np.isclose(0.5 + 0.5 * d, 0.9)   # p_pick_higher = 0.5 + 0.5 D


def test_transitivity_fas_perfect_vs_cycle():
    order = [0, 1, 2]   # best -> worst
    good = [{"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"},
            {"i": 1, "j": 2, "p_util": 0.9, "mode": "logprob"},
            {"i": 0, "j": 2, "p_util": 0.9, "mode": "logprob"}]
    assert np.isclose(transitivity_fas(good, order), 1.0)
    bad = good + [{"i": 2, "j": 0, "p_util": 0.99, "mode": "logprob"}]
    assert transitivity_fas(bad, order) < 1.0


def test_transitivity_triad_detects_cycle():
    # tuples are (p_ab, p_bc, p_ca). Even a clean transitive triple has nonzero soft
    # cycle mass because probabilities multiply (0.95^2*0.05 + 0.05^2*0.95 ~= 0.048).
    transitive = [(0.95, 0.95, 0.05)]   # a>b, b>c, a>c -> consistent
    cyclic = [(0.9, 0.9, 0.9)]          # a>b, b>c, c>a -> 3-cycle, mass ~0.73
    assert transitivity_triad(transitive) > 0.9
    assert transitivity_triad(cyclic) < 0.5


from mu_decisiveness.panel import unidim_fit, reliability, question_robustness


def test_unidim_fit_perfect_model():
    mu = np.array([-2.0, 0.0, 2.0])
    from mu_decisiveness.fit import predict_matrix_caseV
    P = predict_matrix_caseV(mu)
    held = [{"i": 0, "j": 2, "p_util": float(P[0, 2]), "mode": "logprob"}]
    out = unidim_fit(mu, held)
    assert out["brier"] < 1e-6
    assert out["log_loss"] >= 0.0


def test_reliability_position_bias():
    clean = [{"p_fwd": 0.8, "p_rev": 0.2}, {"p_fwd": 0.6, "p_rev": 0.4}]
    out = reliability(clean)
    assert np.isclose(out["order_consistency"], 1.0)
    assert np.isclose(out["position_bias"], 0.0)
    biased = [{"p_fwd": 0.8, "p_rev": 0.5}]   # p_fwd+p_rev-1 = 0.3
    assert reliability(biased)["position_bias"] > 0.0


def test_question_robustness_valence_flip_agreement():
    pairs = [{"p_util_a": 0.9, "p_util_b": 0.88}]   # consistent
    out = question_robustness(pairs)
    assert out["q_agreement"] > 0.95
    assert np.isclose(out["q_sign_agreement"], 1.0)


def _dense_soft_edges(mu_true):
    n = len(mu_true)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = float(0.5 * (1 + math.erf((mu_true[i] - mu_true[j]) / 2.0)))
            rows.append({"i": i, "j": j, "p_util": p, "mode": "logprob"})
    return rows


def test_compute_panel_default_no_bootstrap():
    # default: bootstrap OFF -> finite point estimates, NaN CIs (fast path)
    from mu_decisiveness.panel import compute_panel
    mu_true = np.array([-2.0, -0.7, 0.7, 2.0])
    edges = {"elo": _dense_soft_edges(mu_true), "reverse": [], "triad": [], "cross": []}
    panel = compute_panel(edges, n=4, seed=0)
    for key in ("decisiveness", "transitivity_fas", "unidim_fit_brier"):
        assert np.isfinite(panel[key]["point"])
        assert np.isnan(panel[key]["meas_ci"]).all()
        assert np.isnan(panel[key]["gen_ci"]).all()
    assert 0.0 <= panel["decisiveness"]["point"] <= 1.0


def test_compute_panel_bootstrap_brackets_point():
    # bootstrap ON -> measurement CI brackets the point estimate
    from mu_decisiveness.panel import compute_panel
    mu_true = np.array([-2.0, -0.7, 0.7, 2.0])
    edges = {"elo": _dense_soft_edges(mu_true), "reverse": [], "triad": [], "cross": []}
    panel = compute_panel(edges, n=4, bootstrap=True, B=60, seed=0)
    for key in ("decisiveness", "transitivity_fas"):
        lo, hi = panel[key]["meas_ci"]
        assert lo <= panel[key]["point"] <= hi
    # generalization CI populated for the mu-derived metric
    assert np.isfinite(panel["decisiveness"]["gen_ci"]).all()
