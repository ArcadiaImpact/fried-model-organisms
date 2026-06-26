from __future__ import annotations
import numpy as np
import torch

_NORMAL = torch.distributions.Normal(0.0, 1.0)


def _phi(x: torch.Tensor) -> torch.Tensor:
    return _NORMAL.cdf(x)


def predict_pref_matrix(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    sig_t = torch.as_tensor(sigma, dtype=torch.float64)
    diff = mu_t[:, None] - mu_t[None, :]
    denom = torch.sqrt(sig_t[:, None] ** 2 + sig_t[None, :] ** 2)
    P = _phi(diff / denom)
    P.fill_diagonal_(0.5)
    return P.numpy()


def fit_thurstone(pref: np.ndarray, lr: float = 0.05, steps: int = 2000,
                  test_frac: float = 0.0, l2_sigma: float = 0.01,
                  seed: int = 0) -> dict:
    torch.manual_seed(seed)
    n = pref.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.as_tensor(pref, dtype=torch.float64, device=device)

    # off-diagonal mask. Split the held-out set over UNORDERED pairs {i,j} and
    # mask BOTH directions together: pref is antisymmetric (pref[j,i]=1-pref[i,j]),
    # so training on the mirror (j,i) would leak a held-out (i,j) label otherwise.
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    iu = torch.triu_indices(n, n, offset=1, device=device)          # unordered pairs
    upper = torch.stack([iu[0], iu[1]], dim=1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(upper.shape[0], generator=g)
    n_test_pairs = int(test_frac * upper.shape[0])
    test_pairs = upper[perm[:n_test_pairs]]

    # train_mask: every off-diagonal entry whose unordered pair is NOT held out
    train_mask = mask.clone()
    for a, b in test_pairs.tolist():
        train_mask[a, b] = False
        train_mask[b, a] = False
    # eval over the held-out ordered entries (both directions of held-out pairs)
    if n_test_pairs > 0:
        eval_idx = torch.cat([test_pairs, test_pairs.flip(1)], dim=0)
    else:
        eval_idx = mask.nonzero(as_tuple=False)

    mu = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    log_sigma = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        sigma = torch.exp(log_sigma)
        diff = mu[:, None] - mu[None, :]
        denom = torch.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
        p = _phi(diff / denom).clamp(1e-6, 1 - 1e-6)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        loss = bce[train_mask].mean() + l2_sigma * (log_sigma ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sigma = torch.exp(log_sigma)
        # Gauge fixing: P(x>y) is invariant to scaling all (mu, sigma) by a
        # positive constant and to shifting mu. Center mu (additive gauge) and
        # divide mu, sigma by mean(sigma) (multiplicative gauge) so that the
        # reported magnitudes are data-determined rather than set by l2_sigma.
        scale = sigma.mean()
        mu_c = (mu - mu.mean()) / scale
        sigma_c = sigma / scale
        Phat = predict_pref_matrix(mu_c.cpu().numpy(), sigma_c.cpu().numpy())
        # held-out accuracy: thresholded predicted vs empirical preference
        ph = torch.as_tensor(Phat, device=device)
        pred_label = (ph[eval_idx[:, 0], eval_idx[:, 1]] > 0.5).double()
        emp_label = (target[eval_idx[:, 0], eval_idx[:, 1]] > 0.5).double()
        acc = (pred_label == emp_label).double().mean().item()
        is_heldout = n_test_pairs > 0

    return {
        "mu": mu_c.detach().cpu().numpy(),
        "sigma": sigma_c.detach().cpu().numpy(),
        "test_accuracy": acc,        # held-out when test_frac>0, else train accuracy
        "accuracy_is_heldout": is_heldout,
        "pred_matrix": Phat,
    }
