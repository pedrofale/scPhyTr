"""Linear-time likelihood for Brownian motion on a tree via Felsenstein's pruning.

A homogeneous multivariate BM with trait covariance ``K`` (p x p) induces leaf
values that are jointly Gaussian with covariance ``K ⊗ C``, where ``C`` is the
(dense) n x n phylogenetic covariance. Forming ``C`` is O(n^2) and evaluating the
MVN is O(n^3) -- prohibitive at single-cell scale.

The tree makes the *precision* sparse (a Gaussian Markov random field), so the
marginal likelihood can be computed by Gaussian elimination in post-order
(Felsenstein's pruning) in O(n) time, never forming C. Because every message
covariance is a scalar multiple of the shared trait matrix K, we propagate a
scalar variance ``V`` and a p-vector mean ``m`` per node, and factor K out of the
per-contrast normalization. This decouples the tree (O(n)) from the trait
dimension (O(p^3) for one inverse/determinant of K).
"""

import numpy as np


def _gaussian_contrast_logpdf(diff, var, K_inv, K_logdet):
    """log N(diff; 0, var * K) for a p-vector ``diff`` and scalar ``var``."""
    p = diff.shape[0]
    quad = float(diff @ K_inv @ diff) / var
    return -0.5 * (p * np.log(2.0 * np.pi) + p * np.log(var) + K_logdet + quad)


def bm_pruning_logpdf(tree, trait_means, K):
    """Marginal log-likelihood of leaf traits under homogeneous multivariate BM.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
        Tree whose leaves carry a ``trait`` dict (see ``Tree.set_trait_values``).
    trait_means : array-like, shape (p,)
        Root (ancestral) trait means -- the fixed BM root state.
    K : array-like, shape (p, p)
        Trait covariance (rate) matrix; per-branch increment is N(0, branch * K).

    Returns
    -------
    float
        Log-likelihood, matching the dense ``K ⊗ C`` MVN up to numerical error.
    """
    trait_names = tree.get_trait_names()
    mu = np.asarray(trait_means, dtype=float).ravel()
    K = np.asarray(K, dtype=float)
    K_inv = np.linalg.inv(K)
    sign, K_logdet = np.linalg.slogdet(K)
    if sign <= 0:
        raise ValueError("Trait covariance matrix K must be positive definite.")

    loglik = 0.0

    def descend(node):
        """Return (m, V): node-value distribution N(value; m, V*K) from its subtree.

        Accumulates the contrast contributions of every internal combination in
        the subtree into the enclosing ``loglik``.
        """
        nonlocal loglik

        if node.is_leaf():
            m = np.array([node.trait[name] for name in trait_names], dtype=float)
            return m, 0.0

        children = node.children
        # Seed the accumulator with the first child (branch length folded in).
        m_acc, v_acc = descend(children[0])
        v_acc = v_acc + children[0].dist

        # Sequentially fuse remaining children; each fusion emits one contrast.
        for child in children[1:]:
            m_c, v_c = descend(child)
            v_c = v_c + child.dist

            w = v_acc + v_c
            loglik += _gaussian_contrast_logpdf(m_acc - m_c, w, K_inv, K_logdet)

            v_new = 1.0 / (1.0 / v_acc + 1.0 / v_c)
            m_new = v_new * (m_acc / v_acc + m_c / v_c)
            m_acc, v_acc = m_new, v_new

        return m_acc, v_acc

    m_root, v_root = descend(tree.root)
    # Root's own branch connects the MRCA to the fixed ancestral state mu.
    v_root = v_root + tree.root.dist
    loglik += _gaussian_contrast_logpdf(m_root - mu, v_root, K_inv, K_logdet)

    return loglik


def _ou_branch(alpha, t):
    """OU per-branch contraction ``phi = e^{-alpha t}`` and variance factor
    ``v(t) = (1 - e^{-2 alpha t}) / (2 alpha)``. Stable as alpha -> 0 (v -> t)."""
    phi = np.exp(-alpha * t)
    v = -np.expm1(-2.0 * alpha * t) / (2.0 * alpha)
    return phi, v


def paint_regimes(tree, shift_nodes):
    """Assign an adaptive-regime id to every node by painting shift points.

    The root carries regime 0. Each node in ``shift_nodes`` starts a fresh regime
    that its whole subtree inherits, until a deeper shift overrides it. This is the
    standard "regime painting" used to define OU optima per branch.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    shift_nodes : iterable of ete3 nodes
        Nodes at which the adaptive optimum changes (root is ignored as a shift).

    Returns
    -------
    (regimes, n_regimes) : (dict[node -> int], int)
    """
    shift_set = set(shift_nodes)
    regimes = {}
    next_id = [1]

    def descend(node, current):
        if node is not tree.root and node in shift_set:
            current = next_id[0]
            next_id[0] += 1
        regimes[node] = current
        for child in node.children:
            descend(child, current)

    descend(tree.root, 0)
    return regimes, next_id[0]


def ou_pruning_logpdf(tree, alpha, theta, K, regimes=None, root_value=None):
    """Marginal log-likelihood of leaf traits under multivariate OU.

    Model: scalar mean-reversion ``alpha`` shared across traits, full diffusion
    covariance ``K`` (p x p), and one or more adaptive optima, so the per-branch
    transition is
        z_child | z_parent ~ N(theta_b + e^{-alpha t}(z_parent - theta_b), v(t) K),
    where ``theta_b`` is the optimum of the regime painted on that branch.

    Parameters
    ----------
    alpha : float
        Mean-reversion rate (> 0; for regimes=None and alpha<=0 this delegates to BM).
    theta : array-like
        Single optimum of shape (p,) when ``regimes`` is None, else the optima
        of shape (n_regimes, p).
    K : array-like, shape (p, p)
        Diffusion covariance.
    regimes : dict[node -> int], optional
        Per-node regime ids (see ``paint_regimes``). None means a single global
        optimum (OU-1).
    root_value : array-like, shape (p,), optional
        Fixed ancestral state above the root. Defaults to the root regime's optimum.

    Notes
    -----
    Linear-Gaussian pruning: every message covariance remains a scalar multiple
    of K, so we propagate a scalar variance + p-vector mean per node, fold the
    per-branch contraction into a change of variables, and factor K out of each
    contrast's normalization. O(n) over the tree, O(p^3) for one inverse/logdet.
    """
    if regimes is None and (alpha is None or alpha <= 0):
        return bm_pruning_logpdf(tree, theta, K)
    if alpha is None or alpha <= 0:
        raise ValueError("alpha must be > 0 for multi-regime OU.")

    trait_names = tree.get_trait_names()
    theta_arr = np.asarray(theta, dtype=float)
    if regimes is None:
        thetas = theta_arr.reshape(1, -1)
        regime_of = lambda node: 0
    else:
        thetas = np.atleast_2d(theta_arr)
        regime_of = lambda node: regimes[node]
    p = thetas.shape[1]

    def theta_node(node):
        return thetas[regime_of(node)]

    K = np.asarray(K, dtype=float)
    K_inv = np.linalg.inv(K)
    sign, K_logdet = np.linalg.slogdet(K)
    if sign <= 0:
        raise ValueError("Trait covariance matrix K must be positive definite.")

    a = theta_node(tree.root) if root_value is None else np.asarray(root_value, dtype=float).ravel()

    loglik = 0.0

    def belief(node):
        """Return (m, V): node-value distribution N(value; m, V*K) from the
        subtree below it (before its own parent branch). Accumulates contrasts."""
        nonlocal loglik

        if node.is_leaf():
            m = np.array([node.trait[name] for name in trait_names], dtype=float)
            return m, 0.0

        m_acc, v_acc = message(node.children[0])
        for child in node.children[1:]:
            m_c, v_c = message(child)
            w = v_acc + v_c
            loglik += _gaussian_contrast_logpdf(m_acc - m_c, w, K_inv, K_logdet)
            v_new = 1.0 / (1.0 / v_acc + 1.0 / v_c)
            m_new = v_new * (m_acc / v_acc + m_c / v_c)
            m_acc, v_acc = m_new, v_new
        return m_acc, v_acc

    def message(node):
        """Transform a node's belief through its own branch into a Gaussian in
        the parent's value, N(parent; m, V*K). The contraction induces a change
        of variables whose Jacobian contributes ``p * alpha * t`` to the log-lik."""
        nonlocal loglik
        m_b, v_b = belief(node)
        phi, v = _ou_branch(alpha, node.dist)
        c = (1.0 - phi) * theta_node(node)
        loglik += p * alpha * node.dist  # -p * log(phi)
        return (m_b - c) / phi, (v_b + v) / (phi * phi)

    m_root, v_root = belief(tree.root)
    # Root branch connects the MRCA belief to the fixed ancestral state a.
    phi_r, v_r = _ou_branch(alpha, tree.root.dist)
    mean_root = phi_r * a + (1.0 - phi_r) * theta_node(tree.root)
    loglik += _gaussian_contrast_logpdf(m_root - mean_root, v_root + v_r, K_inv, K_logdet)

    return loglik
