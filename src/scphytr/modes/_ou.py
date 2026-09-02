"""Exact Gaussian likelihood for an OU process on a tree with per-branch optima.

Model (per gene g), for branch b = (parent p -> child c) of length l:

    X_c | X_p ~ N( e^{-a l} X_p + (1 - e^{-a l}) theta_b ,  (s^2 / 2a) (1 - e^{-2 a l}) )
    theta_b = theta_{parent(b)} + delta_b            (optima inherited, changed only by events)

Because the process is linear-Gaussian and Markov on the tree, the tip vector is multivariate
normal. Crucially the **optima enter only the mean**; the covariance depends on (alpha, sigma, tree,
root treatment) alone. So for a gene panel sharing (alpha, sigma) we factorise the covariance ONCE
and reuse it for every gene -- the whole point of a shared-alpha first implementation.

This is "Model 0": the easiest correct implementation (direct covariance construction), against
which any later pruning / message-passing version must be validated.
"""
from __future__ import annotations

import numpy as np

__all__ = ["node_optima", "tip_mean", "tip_cov", "loglik", "ou_decay", "ou_var"]

_TINY = 1e-10


def ou_decay(alpha, t):
    """e^{-alpha t}."""
    return np.exp(-alpha * np.asarray(t, dtype=float))


def ou_var(alpha, sigma2, t):
    """(sigma^2 / 2 alpha) (1 - e^{-2 alpha t}), with the correct Brownian limit sigma^2 t."""
    t = np.asarray(t, dtype=float)
    if alpha < _TINY:
        return sigma2 * t
    return (sigma2 / (2.0 * alpha)) * (-np.expm1(-2.0 * alpha * t))


def node_optima(tree, theta0, delta):
    """Per-node optimum theta (the optimum governing the branch ABOVE that node).

    Parameters
    ----------
    theta0 : (G,) root optimum per gene
    delta  : (n_nodes, G) optimum shifts; ``delta[root]`` is ignored (root uses theta0)

    Returns (n_nodes, G).
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    delta = np.asarray(delta, dtype=float)
    if delta.ndim == 1:
        delta = delta[:, None]
    G = max(theta0.shape[0], delta.shape[1])
    th = np.zeros((tree.n_nodes, G))
    th[0] = theta0
    for v in tree.preorder:
        p = tree.parent[v]
        if p >= 0:
            th[v] = th[p] + delta[v]
    return th


def tip_mean(tree, alpha, theta_nodes, x_root=None):
    """E[X] at the tips, (n_leaves, G). ``x_root`` defaults to the root optimum."""
    theta_nodes = np.asarray(theta_nodes, dtype=float)
    G = theta_nodes.shape[1]
    m = np.zeros((tree.n_nodes, G))
    m[0] = theta_nodes[0] if x_root is None else np.asarray(x_root, dtype=float)
    for v in tree.preorder:
        p = tree.parent[v]
        if p < 0:
            continue
        phi = ou_decay(alpha, tree.dist[v])
        m[v] = phi * m[p] + (1.0 - phi) * theta_nodes[v]
    return m[tree.leaves]


def tip_cov(tree, alpha, sigma, root="stationary"):
    """Cov[X] at the tips, (n_leaves, n_leaves). Shared across genes with the same (alpha, sigma).

    ``root='stationary'`` starts the root at its stationary variance sigma^2/(2 alpha);
    ``root='fixed'`` pins the root to a known value (variance 0).
    """
    sigma2 = float(sigma) ** 2
    if root == "stationary":
        v0 = sigma2 / (2.0 * alpha) if alpha >= _TINY else 0.0
    elif root == "fixed":
        v0 = 0.0
    else:
        raise ValueError("root must be 'stationary' or 'fixed'")

    # node variances by preorder recursion
    v = np.zeros(tree.n_nodes)
    v[0] = v0
    for u in tree.preorder:
        p = tree.parent[u]
        if p >= 0:
            phi2 = ou_decay(alpha, 2.0 * tree.dist[u])
            v[u] = phi2 * v[p] + ou_var(alpha, sigma2, tree.dist[u])

    n = tree.n_leaves
    V = np.zeros((n, n))
    sets = tree.leaf_sets()
    depth = tree.depth
    # pairs whose MRCA is node a live in DIFFERENT child subtrees of a
    for a in range(tree.n_nodes):
        kids = tree.children[a]
        if len(kids) < 2:
            continue
        blocks = []
        for c in kids:
            idx = sets[c]
            leaf_nodes = tree.leaves[idx]
            w = ou_decay(alpha, depth[leaf_nodes] - depth[a])
            blocks.append((idx, w))
        for i in range(len(blocks)):
            ii, wi = blocks[i]
            for j in range(i + 1, len(blocks)):
                jj, wj = blocks[j]
                blk = v[a] * np.outer(wi, wj)
                V[np.ix_(ii, jj)] = blk
                V[np.ix_(jj, ii)] = blk.T
    np.fill_diagonal(V, v[tree.leaves])
    return V


def loglik(Y, tree, alpha, sigma, theta0, delta, sigma_obs=0.0, root="stationary",
           x_root=None, cov=None):
    """Exact log-likelihood of tip data ``Y`` (n_leaves, G), summed over genes.

    ``cov`` optionally supplies a precomputed :func:`tip_cov` (it does not depend on the optima).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    th = node_optima(tree, theta0, delta)
    M = tip_mean(tree, alpha, th, x_root=x_root)
    V = tip_cov(tree, alpha, sigma, root=root) if cov is None else np.array(cov, copy=True)
    if sigma_obs > 0:
        V = V + (sigma_obs ** 2) * np.eye(V.shape[0])
    L = np.linalg.cholesky(V + 1e-10 * np.eye(V.shape[0]))
    R = Y - M
    Z = np.linalg.solve(L, R)
    n, G = Y.shape
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * (G * n * np.log(2 * np.pi) + G * logdet + np.sum(Z * Z)))
