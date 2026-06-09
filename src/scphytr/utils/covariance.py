"""Dense phylogenetic covariances and OU means.

These are O(n^2)-O(n^3) builders used by the Laplace-approximate count likelihood,
where n is the number of leaves (species/clones), which is moderate. They are kept
separate from the O(n) pruning likelihood, which remains the path for large trees.
All quantities are aligned to ``tree.root.get_leaves()`` order.
"""

import numpy as np


def phylo_times(tree):
    """Return (leaf_names, T, S) for a tree.

    T[i] = root-to-tip time of leaf i (including the root branch);
    S[i, j] = shared time from the ancestral state down to MRCA(i, j)
    (including the root branch). S has T on its diagonal.
    """
    root = tree.root
    leaves = root.get_leaves()
    n = len(leaves)

    depth = {root: 0.0}
    for node in root.traverse("preorder"):
        for child in node.children:
            depth[child] = depth[node] + child.dist

    T = np.array([root.dist + depth[leaf] for leaf in leaves])
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                S[i, j] = T[i]
            else:
                mrca = tree.phylotree.get_common_ancestor(leaves[i], leaves[j])
                S[i, j] = root.dist + depth[mrca]
    return [leaf.name for leaf in leaves], T, S


def bm_covariance(tree):
    """BM phylogenetic covariance (shared evolutionary time), unit rate."""
    _, _, S = phylo_times(tree)
    return S


def ou_covariance(tree, alpha):
    """OU phylogenetic covariance with scalar mean-reversion, unit diffusion.

    C[i, j] = e^{-alpha (T_i + T_j - 2 S_ij)} (1 - e^{-2 alpha S_ij}) / (2 alpha).
    """
    _, T, S = phylo_times(tree)
    return np.exp(-alpha * (T[:, None] + T[None, :] - 2.0 * S)) * (1.0 - np.exp(-2.0 * alpha * S)) / (2.0 * alpha)


def ou_regime_mean(tree, alpha, thetas, regimes, root_value):
    """Per-leaf OU mean under regime-specific optima (deterministic recursion).

    E[child] = e^{-alpha t} E[parent] + (1 - e^{-alpha t}) theta_{regime(child)},
    seeded above the root by the fixed ancestral state ``root_value``.
    """
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    root = tree.root
    a = float(np.asarray(root_value).ravel()[0])
    phi_r = np.exp(-alpha * root.dist)
    E = {root: phi_r * a + (1.0 - phi_r) * thetas[regimes[root]]}
    for node in root.traverse("preorder"):
        for child in node.children:
            phi = np.exp(-alpha * child.dist)
            E[child] = phi * E[node] + (1.0 - phi) * thetas[regimes[child]]
    return np.array([E[leaf] for leaf in root.get_leaves()])
