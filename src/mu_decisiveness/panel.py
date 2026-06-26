from __future__ import annotations

import math
import numpy as np

from .fit import predict_matrix_caseV


def decisiveness(mu) -> float:
    """mean|2 Phi - 1| over unordered pairs of the fitted Case V matrix. Bounded [0,1]."""
    P = predict_matrix_caseV(mu)
    iu = np.triu_indices(P.shape[0], k=1)
    return float(np.mean(np.abs(2 * P[iu] - 1)))


def decisiveness_raw(rows) -> float:
    """mean|2 p_util - 1| over observed edges (resolution-limited diagnostic)."""
    p = np.array([float(r["p_util"]) for r in rows])
    return float(np.mean(np.abs(2 * p - 1))) if len(p) else float("nan")


def transitivity_fas(rows, order) -> float:
    """1 - confidence-weighted fraction of edges pointing backward vs `order`
    (best->worst). Weight = |2 p_util - 1|."""
    rank = {item: r for r, item in enumerate(order)}
    num = den = 0.0
    for r in rows:
        i, j, p = r["i"], r["j"], float(r["p_util"])
        if i not in rank or j not in rank:
            continue
        w = abs(2 * p - 1)
        prefers_i = p > 0.5
        i_before_j = rank[i] < rank[j]   # i ranked better
        backward = prefers_i != i_before_j
        den += w
        if backward:
            num += w
    return 1.0 - (num / den) if den else float("nan")


def transitivity_triad(triads) -> float:
    """triads: list of (p_ab, p_bc, p_ca). Returns 1 - mean soft cycle mass."""
    if not triads:
        return float("nan")
    masses = []
    for p_ab, p_bc, p_ca in triads:
        fwd = p_ab * p_bc * p_ca
        bwd = (1 - p_ab) * (1 - p_bc) * (1 - p_ca)
        masses.append(fwd + bwd)
    return 1.0 - float(np.mean(masses))


def unidim_fit(mu, held_rows) -> dict:
    """Brier + log-loss of `mu` evaluated on `held_rows` (pass a held-out edge split for generalization). Lower is better.
    noise_floor = irreducible Binomial variance for sample edges (else 0)."""
    P = predict_matrix_caseV(mu)
    briers, lls, floors = [], [], []
    for r in held_rows:
        y = float(r["p_util"])
        phat = float(np.clip(P[r["i"], r["j"]], 1e-9, 1 - 1e-9))
        briers.append((phat - y) ** 2)
        lls.append(-(y * np.log(phat) + (1 - y) * np.log(1 - phat)))
        if r.get("mode") == "sample":
            N = float(r["wins_i"] + r["wins_j"]) or 1.0
            floors.append(y * (1 - y) / N)
        else:
            floors.append(0.0)
    return {
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("nan"),
        "noise_floor": float(np.mean(floors)) if floors else 0.0,
    }


def unidim_r2(mu, rows) -> float:
    """Deviance R² — goodness-of-fit of the single-latent-dimension (Case-V) model, decoupled
    from decisiveness. Fraction of the *explainable* preference signal that one μ axis captures:

        R² = (log2 − CE_fit) / (log2 − H)
           = 1 − KL(observed ‖ model) / (log2 − H)

    where per edge CE_fit = −p logP̂ −(1−p)log(1−P̂) is the model's cross-entropy against the
    observed choice prob p, H = −p log p −(1−p)log(1−p) is the binary entropy of the observed
    prob (the SATURATED model — irreducible), and log2 is the indifferent-model cross-entropy.

    =1 when the 1-D model reproduces the data exactly; =0 when no better than chance; <0 worse.
    The denominator (log2 − H) is the explainable signal, so decisiveness cancels: a weakly- and a
    strongly-decisive model that are both perfectly 1-D both score ≈1; a decisive-but-multidimensional
    model scores low. Exact for logprob edges (p is the true choice prob, no sampling noise — and the
    MLE minimizes CE_fit, so this is the residual after the best possible 1-D fit). For sample-mode p̂
    is noisy so H is slightly underestimated (mild upward R² bias at small N)."""
    P = predict_matrix_caseV(mu)
    LOG2 = math.log(2.0)
    ce, h = [], []
    for r in rows:
        p = float(np.clip(r["p_util"], 1e-12, 1 - 1e-12))
        phat = float(np.clip(P[r["i"], r["j"]], 1e-12, 1 - 1e-12))
        ce.append(-(p * math.log(phat) + (1 - p) * math.log(1 - phat)))
        h.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
    if not ce:
        return float("nan")
    denom = LOG2 - float(np.mean(h))
    if denom <= 1e-6:                      # ~no signal to explain → R² ill-defined
        return float("nan")
    return float((LOG2 - float(np.mean(ce))) / denom)


def reliability(reverse_pairs) -> dict:
    """reverse_pairs: list of {p_fwd, p_rev} where p_fwd=P(pick i|i first),
    p_rev=P(pick i|i second). order_consistency=1-mean|p_fwd+p_rev-1|,
    position_bias=mean(p_fwd+p_rev-1): signed net calibration excess across both orderings (symmetric first-vs-second slot bias cancels and is NOT captured here)."""
    if not reverse_pairs:
        return {"order_consistency": float("nan"), "position_bias": float("nan")}
    diffs = np.array([p["p_fwd"] + p["p_rev"] - 1.0 for p in reverse_pairs])
    return {"order_consistency": float(1.0 - np.mean(np.abs(diffs))),
            "position_bias": float(np.mean(diffs))}


def question_robustness(cross_pairs) -> dict:
    """cross_pairs: list of {p_util_a, p_util_b} for the same item pair under two
    questions (both oriented to P(item_i > item_j)).

    q_agreement is the Pearson correlation of the two framings' p_util across pairs.
    This is the DECISIVENESS-ROBUST form: the older 1-mean|a-b| version is mechanically
    inflated for near-indifferent models (a,b both ~0.5 -> tiny |a-b| -> looks "consistent"),
    giving a spurious -0.8 correlation with decisiveness across models; the correlation form
    normalizes each model by its own variance and drops that confound to ~-0.1.
    (Cross-framing disagreement scales with opinion strength, so variance-normalization is the
    right fix here -- unlike order_consistency, where position bias is an additive offset that
    is already decisiveness-independent, so its 1-mean|.| form is kept.)
    q_agreement_absdiff (old metric) and q_sign_agreement are retained as diagnostics.
    For singleton or zero-variance inputs, Pearson is undefined, so q_agreement falls
    back to q_agreement_absdiff."""
    if not cross_pairs:
        return {"q_agreement": float("nan"), "q_agreement_absdiff": float("nan"),
                "q_sign_agreement": float("nan")}
    a = np.array([p["p_util_a"] for p in cross_pairs])
    b = np.array([p["p_util_b"] for p in cross_pairs])
    absdiff = 1.0 - np.mean(np.abs(a - b))
    sign = np.mean(np.sign(a - 0.5) == np.sign(b - 0.5))
    corr = (float(np.corrcoef(a, b)[0, 1])
            if len(a) >= 2 and a.std() > 1e-9 and b.std() > 1e-9 else float(absdiff))
    return {"q_agreement": corr, "q_agreement_absdiff": float(absdiff),
            "q_sign_agreement": float(sign)}


from .fit import fit_caseV_mle, bootstrap_measurement, bootstrap_items


def _ci(samples, lo=2.5, hi=97.5):
    s = np.asarray([x for x in samples if np.isfinite(x)])
    if s.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(s, lo)), float(np.percentile(s, hi))]


def _bootstrap_raw(items, metric_fn, B, seed):
    """Percentile CI by resampling a list of observations with replacement."""
    rng = np.random.default_rng(seed)
    if not items:
        return [float("nan"), float("nan")]
    arr = np.array(items, dtype=object)
    vals = []
    for _ in range(B):
        sel = rng.integers(0, len(arr), size=len(arr))
        vals.append(metric_fn(list(arr[sel])))
    return _ci(vals)


def _nan_ci():
    return [float("nan"), float("nan")]


def compute_panel(edges_by_phase, n, bootstrap=False, B=200, seed=0, fit_steps=2000,
                  mu_init=None, primary_qid=None):
    """Compute the metric panel. Point estimates are always computed (one Case V fit).

    Bootstrap CIs are OFF by default (fast point-estimate runs). Pass bootstrap=True to
    populate measurement + generalization CIs; the bootstrap fits are vectorized on GPU
    and warm-started from the point estimate, so opting in is cheap relative to before.
    When bootstrap is off, every `meas_ci`/`gen_ci` is [nan, nan].

    primary_qid: if set, the mu fit (and the mu-derived metrics + transitivity_fas) use
    only elo edges with this question_id. Use it to recompute legacy runs whose elo phase
    mixed framings; new runs already elo-sample the primary question only, so it's a no-op.
    mu_init: optional warm-start for the point estimate fit.
    """
    elo = edges_by_phase.get("elo", [])
    if primary_qid is not None:
        elo = [e for e in elo if e.get("question_id") == primary_qid]
    mu = fit_caseV_mle(elo, n=n, seed=seed, steps=fit_steps, mu_init=mu_init)["mu"]
    order = list(np.argsort(-mu))           # best -> worst

    def meas_fit(metric_fn):                # mu-derived measurement CI (warm-started)
        return _ci(bootstrap_measurement(elo, n, B, metric_fn, seed, mu_init=mu)) \
            if bootstrap else _nan_ci()

    def gen_fit(metric_fn):                 # mu-derived generalization CI (item cluster)
        return _ci(bootstrap_items(elo, n, B, metric_fn, seed, mu_init=mu)) \
            if bootstrap else _nan_ci()

    def meas_raw(items, metric_fn):         # raw-graph measurement CI
        return _bootstrap_raw(items, metric_fn, B, seed) if bootstrap else _nan_ci()

    panel = {}
    panel["decisiveness"] = {"point": decisiveness(mu),
                             "meas_ci": meas_fit(decisiveness), "gen_ci": gen_fit(decisiveness)}
    panel["decisiveness_raw"] = {"point": decisiveness_raw(elo),
                                 "meas_ci": meas_raw(elo, decisiveness_raw), "gen_ci": _nan_ci()}

    # --- unidimensional fit: HELD-OUT split of elo edges (fit on train, eval on test) ---
    m = len(elo)
    if m >= 5:
        rng_uf = np.random.default_rng(seed)
        perm = rng_uf.permutation(m)
        n_test = max(1, int(0.2 * m))
        test_set = set(perm[:n_test].tolist())
        train_e = [e for k, e in enumerate(elo) if k not in test_set]
        test_e = [e for k, e in enumerate(elo) if k in test_set]
        mu_tr = fit_caseV_mle(train_e, n=n, seed=seed, steps=fit_steps, mu_init=mu)["mu"]
        uf = unidim_fit(mu_tr, test_e)
        panel["unidim_fit_brier"] = {
            "point": uf["brier"],
            "meas_ci": meas_raw(test_e, lambda r: unidim_fit(mu_tr, r)["brier"]),
            "gen_ci": _nan_ci()}
        panel["unidim_fit_log_loss"] = {
            "point": uf["log_loss"],
            "meas_ci": meas_raw(test_e, lambda r: unidim_fit(mu_tr, r)["log_loss"]),
            "gen_ci": _nan_ci()}
    else:
        panel["unidim_fit_brier"] = {"point": float("nan"), "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
        panel["unidim_fit_log_loss"] = {"point": float("nan"), "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}

    # --- raw-graph metrics ---
    panel["transitivity_fas"] = {"point": transitivity_fas(elo, order),
                                 "meas_ci": meas_raw(elo, lambda r: transitivity_fas(r, order)),
                                 "gen_ci": _nan_ci()}
    triads = edges_by_phase.get("triad", [])
    panel["transitivity_triad"] = {"point": transitivity_triad(triads),
                                   "meas_ci": meas_raw(triads, transitivity_triad), "gen_ci": _nan_ci()}
    rev = edges_by_phase.get("reverse", [])
    panel["order_consistency"] = {"point": reliability(rev)["order_consistency"],
                                  "meas_ci": meas_raw(rev, lambda r: reliability(r)["order_consistency"]),
                                  "gen_ci": _nan_ci()}
    cross = edges_by_phase.get("cross", [])
    qr = question_robustness(cross)
    panel["q_agreement"] = {"point": qr["q_agreement"],
                            "meas_ci": meas_raw(cross, lambda r: question_robustness(r)["q_agreement"]),
                            "gen_ci": _nan_ci()}
    # diagnostics: old decisiveness-confounded abs-diff form + sign agreement
    panel["q_agreement_absdiff"] = {"point": qr["q_agreement_absdiff"],
                                    "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
    panel["q_sign_agreement"] = {"point": qr["q_sign_agreement"],
                                 "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
    panel["mu_std_diagnostic"] = {"point": float(np.std(mu)),
                                  "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
    return panel
