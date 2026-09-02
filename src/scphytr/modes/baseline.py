"""A faithful, self-contained re-implementation of SCOUT's decision rule, in our own likelihood.

SCOUT fits, INDEPENDENTLY PER GENE, three hypotheses on a given regime painting and picks the
minimum-AICc winner:

    BM1  neutral drift                     params: sigma^2, root state              (k = 2)
    OU1  one global optimum                params: alpha, sigma^2, theta            (k = 3)
    OUx  one optimum per regime            params: alpha, sigma^2, theta_1..theta_x (k = 2 + x)

We reproduce that rule exactly, but compute the likelihood with our own exact OU machinery, so the
comparison later is about MODEL STRUCTURE (per-gene vs pooled-across-genes), not about who wrote a
better optimiser.

Two exact simplifications make this fast and robust (no multi-dimensional Nelder-Mead):
  * the tip mean is LINEAR in the optima -> profile theta out by generalised least squares;
  * the covariance factorises as sigma^2 * R(alpha) -> profile sigma^2 out analytically.
So only alpha is optimised numerically, on a 1-D grid, and R(alpha) is factorised ONCE per alpha and
reused across every gene.
"""
from __future__ import annotations

import numpy as np

from ._ou import ou_decay, tip_cov

__all__ = ["paint_regimes", "regime_design", "fit_models", "classify_genes", "MODELS"]

MODELS = ("BM1", "OU1", "OUX")


def paint_regimes(tree, leaf_regime):
    """Assign a regime to every node from leaf labels by simple parsimony-style downward painting.

    SCOUT uses ape::ace (equal-rates Mk, max-likelihood state). We use the cheaper deterministic
    rule: a node takes the unique regime of its descendant leaves when they agree, else the root
    regime. On the simulated trees here (clades are regime-coherent by construction) this coincides
    with the ML painting; swap in a proper Mk reconstruction for real data.
    """
    leaf_regime = np.asarray(leaf_regime)
    uniq = list(dict.fromkeys(leaf_regime.tolist()))
    code = {r: i for i, r in enumerate(uniq)}
    sets = tree.leaf_sets()
    node_regime = np.zeros(tree.n_nodes, dtype=int)
    root_rs = {code[r] for r in leaf_regime[sets[0]]}
    root_code = min(root_rs)
    for v in range(tree.n_nodes):
        rs = {code[r] for r in leaf_regime[sets[v]]}
        node_regime[v] = rs.pop() if len(rs) == 1 else root_code
    return node_regime, uniq


def regime_design(tree, alpha, node_regime, n_regimes):
    """Design matrix W (n_leaves, n_regimes) with tip_mean = W @ theta.

    The tip mean is linear in the optima, so each column is the tip mean obtained by setting one
    regime's optimum to 1 and the rest to 0 (root state = its own regime's optimum).
    """
    W = np.zeros((tree.n_leaves, n_regimes))
    for r in range(n_regimes):
        th = (node_regime == r).astype(float)
        m = np.zeros(tree.n_nodes)
        m[0] = th[0]
        for v in tree.preorder:
            p = tree.parent[v]
            if p < 0:
                continue
            phi = ou_decay(alpha, tree.dist[v])
            m[v] = phi * m[p] + (1.0 - phi) * th[v]
        W[:, r] = m[tree.leaves]
    return W


def _profile(Y, R, W):
    """GLS profile over theta and sigma^2. Returns (loglik per gene, theta, sigma2)."""
    n, G = Y.shape
    L = np.linalg.cholesky(R + 1e-10 * np.eye(n))
    Ry = np.linalg.solve(L, Y)                        # (n, G)
    Rw = np.linalg.solve(L, W)                        # (n, p)
    A = Rw.T @ Rw
    theta = np.linalg.solve(A + 1e-12 * np.eye(A.shape[0]), Rw.T @ Ry)   # (p, G)
    resid = Ry - Rw @ theta
    rss = np.sum(resid * resid, axis=0)               # (G,)
    sigma2 = np.maximum(rss / n, 1e-12)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    ll = -0.5 * (n * np.log(2 * np.pi) + n * np.log(sigma2) + logdet + n)
    return ll, theta, sigma2


def _aicc(ll, k, n):
    aic = 2 * k - 2 * ll
    denom = max(n - k - 1, 1)
    return aic + (2 * k * (k + 1)) / denom


def fit_models(Y, tree, node_regime=None, n_regimes=1, alpha_grid=None, models=MODELS):
    """Fit each model to every gene. Returns {model: dict(loglik, aicc, alpha, sigma2, theta)}."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, G = Y.shape
    if alpha_grid is None:
        alpha_grid = np.exp(np.linspace(np.log(0.02), np.log(20.0), 24))
    out = {}

    if "BM1" in models:
        R = tip_cov(tree, 1e-12, 1.0, root="fixed")
        W = np.ones((n, 1))
        ll, th, s2 = _profile(Y, R, W)
        out["BM1"] = dict(loglik=ll, aicc=_aicc(ll, 2, n), alpha=np.zeros(G),
                          sigma2=s2, theta=th)

    for name, nreg in (("OU1", 1), ("OUX", n_regimes)):
        if name not in models or (name == "OUX" and n_regimes < 2):
            continue
        best = None
        for a in alpha_grid:
            R = tip_cov(tree, a, 1.0, root="stationary")
            W = (np.ones((n, 1)) if nreg == 1
                 else regime_design(tree, a, node_regime, n_regimes))
            ll, th, s2 = _profile(Y, R, W)
            if best is None:
                best = dict(loglik=ll, alpha=np.full(G, a), sigma2=s2, theta=th)
            else:
                better = ll > best["loglik"]
                best["alpha"] = np.where(better, a, best["alpha"])
                best["sigma2"] = np.where(better, s2, best["sigma2"])
                best["theta"] = np.where(better[None, :], th, best["theta"])
                best["loglik"] = np.where(better, ll, best["loglik"])
        k = 2 + nreg
        best["aicc"] = _aicc(best["loglik"], k, n)
        out[name] = best
    return out


def classify_genes(Y, tree, leaf_regime=None, alpha_grid=None, min_alpha=0.0, delta_aicc=0.0):
    """SCOUT's rule: per gene, the minimum-AICc model among BM1 / OU1 / OUX.

    ``min_alpha`` mimics SCOUT's real-data filter (drop OU fits with tiny alpha as indistinguishable
    from BM); ``delta_aicc`` mimics their optional stringency margin.
    """
    if leaf_regime is None:
        node_regime, uniq = None, [0]
    else:
        node_regime, uniq = paint_regimes(tree, leaf_regime)
    fits = fit_models(Y, tree, node_regime=node_regime, n_regimes=len(uniq),
                      alpha_grid=alpha_grid)
    names = [m for m in MODELS if m in fits]
    A = np.stack([fits[m]["aicc"] for m in names])        # (n_models, G)
    order = np.argsort(A, axis=0)
    best = order[0]
    call = np.array([names[i] for i in best], dtype=object)
    if A.shape[0] > 1:
        gap = np.take_along_axis(A, order[1:2], axis=0)[0] - np.take_along_axis(A, order[0:1], axis=0)[0]
    else:
        gap = np.full(A.shape[1], np.inf)
    if min_alpha > 0:
        for i, m in enumerate(call):
            if m in ("OU1", "OUX") and fits[m]["alpha"][i] < min_alpha:
                call[i] = "BM1"
    if delta_aicc > 0:
        call = np.where(gap >= delta_aicc, call, "ambiguous")
    return call, fits, gap
