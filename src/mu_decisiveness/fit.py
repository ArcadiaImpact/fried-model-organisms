from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
_SQRT2 = math.sqrt(2.0)


def _phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def load_mu_init(prev_mu, items) -> Optional[np.ndarray]:
    """Align a prior {item: mu} mapping or mu.json path to `items` order."""
    if prev_mu is None:
        return None
    if isinstance(prev_mu, (str, Path)):
        prev_mu = json.loads(Path(prev_mu).read_text())

    arr = np.zeros(len(items), dtype=np.float64)
    overlap = 0
    for k, item in enumerate(items):
        if item in prev_mu:
            arr[k] = float(prev_mu[item])
            overlap += 1

    frac = overlap / len(items) if items else 1.0
    if frac < 0.5:
        logger.warning("Low mu_init overlap fraction: %.3f (%d/%d items)", frac, overlap, len(items))
    return arr


def normalize_edges(rows):
    """rows -> (i_idx, j_idx, w_pos, w_neg) as numpy arrays. See contract in plan."""
    i_idx, j_idx, w_pos, w_neg = [], [], [], []
    for r in rows:
        i_idx.append(int(r["i"]))
        j_idx.append(int(r["j"]))
        if r.get("mode") == "sample":
            w_pos.append(float(r["wins_i"]))
            w_neg.append(float(r["wins_j"]))
        else:
            p = float(r["p_util"])
            w_pos.append(p)
            w_neg.append(1.0 - p)
    return (np.asarray(i_idx), np.asarray(j_idx),
            np.asarray(w_pos, dtype=np.float64), np.asarray(w_neg, dtype=np.float64))


def predict_matrix_caseV(mu) -> np.ndarray:
    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float64)
    diff = mu_t[:, None] - mu_t[None, :]
    P = _phi(diff / _SQRT2)              # P_ij = Phi((mu_i-mu_j)/sqrt2), Case V with sigma=1
    P.fill_diagonal_(0.5)
    return P.numpy()


def fit_caseV_mle(rows, n, steps=2000, lr=0.05, seed=0, device=None, mu_init=None) -> dict:
    """Homoscedastic Thurstone Case V MLE on mu (sigma fixed=1). No prior.
    P(item_i > item_j) = Phi((mu_i - mu_j)/sqrt2). Gauge: center mu.
    mu_init: optional (n,) warm-start (fewer steps needed near a known solution)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    ii = torch.as_tensor(i_idx, device=device)
    jj = torch.as_tensor(j_idx, device=device)
    wp = torch.as_tensor(w_pos, device=device)
    wn = torch.as_tensor(w_neg, device=device)
    if mu_init is None:
        mu = torch.zeros(n, dtype=torch.float64, device=device)
    else:
        mu = torch.as_tensor(np.asarray(mu_init), dtype=torch.float64, device=device).clone()
    mu.requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P = _phi((mu[ii] - mu[jj]) / _SQRT2).clamp(1e-9, 1 - 1e-9)
        nll = -(wp * torch.log(P) + wn * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu_c = mu - mu.mean()            # additive gauge
    return {"mu": mu_c.detach().cpu().numpy()}


def _batched_caseV_fit(ii_be, jj_be, wp_be, wn_be, n, steps, lr, device, mu_init=None):
    """Fit B replicates simultaneously as one (B, n) mu tensor on `device`.
    ii_be/jj_be/wp_be/wn_be are (B, E) tensors (indices may be an expanded view).
    mu_init: optional (n,) or (B, n) warm-start. Returns centered mu as (B, n) numpy."""
    B = wp_be.shape[0]
    if mu_init is None:
        mu = torch.zeros(B, n, dtype=torch.float64, device=device)
    else:
        mi = torch.as_tensor(np.asarray(mu_init), dtype=torch.float64, device=device)
        mu = (mi.unsqueeze(0).expand(B, n).clone() if mi.ndim == 1 else mi.clone())
    mu.requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=lr)
    bidx = torch.arange(B, device=device).unsqueeze(1)            # (B, 1)
    for _ in range(steps):
        opt.zero_grad()
        diff = (mu[bidx, ii_be] - mu[bidx, jj_be]) / _SQRT2       # (B, E)
        P = _phi(diff).clamp(1e-9, 1 - 1e-9)
        nll = -(wp_be * torch.log(P) + wn_be * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        return (mu - mu.mean(dim=1, keepdim=True)).detach().cpu().numpy()


def bootstrap_measurement(rows, n, B, metric_fn, seed=0, steps=600, lr=0.05,
                          device=None, mu_init=None):
    """Measurement bootstrap: resample edges with replacement (and sample-mode draws),
    fit all B replicates in ONE batched GPU optimization. Warm-start from mu_init
    (the point estimate) to converge in far fewer steps."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    is_sample = np.array([r.get("mode") == "sample" for r in rows])
    totals = w_pos + w_neg                            # N for sample, 1.0 for soft
    E = len(rows)

    pick = rng.integers(0, E, size=(B, E))            # (B, E) per-replicate edge resample
    wp = w_pos[pick].copy()
    wn = w_neg[pick].copy()
    smask = is_sample[pick]
    if smask.any():
        Ns = totals[pick]
        ps = np.where(Ns > 0, wp / np.maximum(Ns, 1e-9), 0.5)
        draws = rng.binomial(np.where(smask, Ns, 0).astype(int), np.clip(ps, 0, 1))
        wp = np.where(smask, draws, wp)
        wn = np.where(smask, Ns - draws, wn)

    ii_be = torch.as_tensor(i_idx[pick], device=device)
    jj_be = torch.as_tensor(j_idx[pick], device=device)
    wp_be = torch.as_tensor(wp, device=device, dtype=torch.float64)
    wn_be = torch.as_tensor(wn, device=device, dtype=torch.float64)
    mu_b = _batched_caseV_fit(ii_be, jj_be, wp_be, wn_be, n, steps, lr, device, mu_init)
    return np.array([metric_fn(mu_b[b]) for b in range(B)])


def bootstrap_items(rows, n, B, metric_fn, seed=0, steps=400, lr=0.05,
                    device=None, mu_init=None):
    """Cluster bootstrap over items, vectorized. Drawing n items uniformly with
    replacement makes the induced multigraph weight each original edge (oi,oj) by
    count[oi]*count[oj] (counts ~ Multinomial(n, 1/n)). So instead of materializing the
    ragged multigraph and refitting per replicate, we weight each edge's likelihood by
    that product and fit all B replicates in ONE batched GPU optimization. Items with
    count 0 contribute no edges and keep their warm-start (mu_init) value."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    E = len(rows)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)   # (B, n)
    w_items = counts[:, i_idx] * counts[:, j_idx]                                  # (B, E)
    ii_be = torch.as_tensor(i_idx, device=device).unsqueeze(0).expand(B, E)
    jj_be = torch.as_tensor(j_idx, device=device).unsqueeze(0).expand(B, E)
    wi = torch.as_tensor(w_items, device=device, dtype=torch.float64)
    wp_be = torch.as_tensor(w_pos, device=device, dtype=torch.float64).unsqueeze(0) * wi
    wn_be = torch.as_tensor(w_neg, device=device, dtype=torch.float64).unsqueeze(0) * wi
    mu_b = _batched_caseV_fit(ii_be, jj_be, wp_be, wn_be, n, steps, lr, device, mu_init)
    return np.array([metric_fn(mu_b[b]) for b in range(B)])
