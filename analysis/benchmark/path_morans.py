"""PATH-style phylogenetic autocorrelation (Moran's I) for benchmarking.

PATH (Schiffman et al. 2024) quantifies cell-state heritability as the phylogenetic
autocorrelation of a trait across the tree -- Moran's I with a phylogenetic weight
matrix. This is a compact, faithful Python implementation for the heritability
benchmark: ``morans_I`` computes the statistic, ``path_test`` adds a permutation
p-value. Like EvoGeneX (and unlike scPhyTr) it operates on the *normalized trait*,
not raw counts, so it is expected to degrade as sequencing depth drops.
"""
import numpy as np


def phylo_weights(tree, leaf_names):
    """Shared-ancestry weight matrix W (off-diagonal): W_ij = root-to-MRCA(i,j) time."""
    leaves = tree.phylotree.get_leaves()
    name_idx = {l.name: k for k, l in enumerate(leaves)}
    order = [name_idx[n] for n in leaf_names]
    n = len(leaves)
    # depth (root-to-node distance) of every node
    depth = {}
    for nd in tree.phylotree.traverse("preorder"):
        depth[nd] = 0.0 if nd.up is None else depth[nd.up] + nd.dist
    # leaves under each internal node -> fill shared depth for all pairs
    C = np.zeros((n, n))
    for nd in tree.phylotree.traverse("postorder"):
        lv = [name_idx[l.name] for l in nd.get_leaves()]
        if len(lv) > 1:
            d = depth[nd]
            for a in lv:
                for b in lv:
                    if a != b and C[a, b] == 0.0:
                        C[a, b] = d
    C = C[np.ix_(order, order)]
    np.fill_diagonal(C, 0.0)
    return C


def morans_I(x, W):
    x = np.asarray(x, float)
    x = x - x.mean()
    S0 = W.sum()
    den = np.sum(x * x)
    if den <= 0 or S0 <= 0:
        return 0.0
    return (len(x) / S0) * (x @ W @ x) / den


def path_test(x, W, n_perm=499, rng=None, alternative="greater"):
    """Moran's I + permutation p-value (heritability test)."""
    rng = rng or np.random.default_rng(0)
    I_obs = morans_I(x, W)
    null = np.array([morans_I(rng.permutation(x), W) for _ in range(n_perm)])
    p = (1 + np.sum(null >= I_obs)) / (n_perm + 1)
    return I_obs, p
