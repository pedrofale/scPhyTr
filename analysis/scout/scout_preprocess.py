"""SCOUT's preprocessing, transcribed from their R so our comparison is like-for-like.

Reference: `lineage_smooth` and `formatSCOUT` in SCOUT/R/SCOUT_EM_utils.R (lines ~676 and ~1055).

Their pipeline, in order:
  1. tree hygiene   -- missing edge lengths -> 1; ZERO-length edges -> 1e-7
  2. normalisation  -- ``log1p(counts)``  (note: plain log1p of RAW counts, no CPM/size-factor step)
  3. lineage smoothing (optional, parameter ``k``) -- a Gaussian kernel on PATRISTIC distance whose
     bandwidth is each cell's distance to its k-th nearest lineage neighbour
  4. floor          -- optionally replace exact zeros with a small constant
  5. optional tree height scaling to 1

Two R semantics are load-bearing and easy to get wrong when porting, so they are reproduced
deliberately here:
  * ``matrix / vector`` in R recycles COLUMN-MAJOR, so ``adj^2 / sigmas`` divides ROW i by
    ``sigmas[i]`` (not column j by sigmas[j]);
  * the kernel divides by ``sigma``, NOT ``sigma^2``;
  * the affinity matrix is symmetrised (``A + A'``, which doubles the diagonal), THEN masked by row
    i's own window, THEN row-normalised, and only THEN is the diagonal overwritten with 1 -- so the
    rows of the final operator do not sum to 1.
"""
from __future__ import annotations

import numpy as np

__all__ = ["patristic", "lineage_smooth", "log_normalize", "scout_preprocess", "clean_edges"]

ZERO_EDGE_REPLACEMENT = 1e-7


def clean_edges(tree, zero_replacement=ZERO_EDGE_REPLACEMENT, inplace=True):
    """SCOUT's tree hygiene: zero-length edges become ``1e-7`` (missing lengths are already 1)."""
    t = tree if inplace else _copy_tree(tree)
    d = np.array(t.dist, dtype=float)
    root = t.parent < 0
    bad = (d <= 0) & (~root)
    if bad.any():
        d[bad] = zero_replacement
        t.dist = d
        depth = np.zeros(t.n_nodes)
        for v in t.preorder:
            p = t.parent[v]
            if p >= 0:
                depth[v] = depth[p] + t.dist[v]
        t.depth = depth
    return t


def _copy_tree(tree):
    from scphytr.modes._tree import Tree
    return Tree(tree.parent.copy(), tree.dist.copy(), list(tree.name),
                [list(c) for c in tree.children])


def patristic(tree):
    """Tip-by-tip patristic distance (R's ``ape::cophenetic``): d_ij = depth_i + depth_j - 2*depth_mrca."""
    n = tree.n_leaves
    depth = tree.depth
    leaf_depth = depth[tree.leaves]
    D = np.add.outer(leaf_depth, leaf_depth)
    sets = tree.leaf_sets()
    mrca_depth = np.zeros((n, n))
    for a in range(tree.n_nodes):          # pairs whose MRCA is a live in different child subtrees
        kids = tree.children[a]
        if len(kids) < 2:
            continue
        for i in range(len(kids)):
            ii = sets[kids[i]]
            for j in range(i + 1, len(kids)):
                jj = sets[kids[j]]
                mrca_depth[np.ix_(ii, jj)] = depth[a]
                mrca_depth[np.ix_(jj, ii)] = depth[a]
    np.fill_diagonal(mrca_depth, leaf_depth)
    D = D - 2.0 * mrca_depth
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def lineage_smooth(tree, X, k, s=None, distances=None):
    """SCOUT's ``lineage_smooth``: kernel smoothing of tip values over lineage distance.

    ``X`` is (n_leaves, n_genes) in tree tip order; ``k`` is the neighbourhood size (their
    ``smoothing_k``, default 8). ``s`` optionally fixes the bandwidth instead of the adaptive one.
    """
    X = np.asarray(X, dtype=float)
    D = patristic(tree) if distances is None else np.asarray(distances, dtype=float)
    m = D.shape[0]
    if X.shape[0] != m:
        raise ValueError(f"X has {X.shape[0]} rows but the tree has {m} tips")

    # bandwidth = distance to the k-th nearest OTHER tip (sort includes self at position 0)
    order = np.sort(D, axis=1)
    kk = min(k, m - 1)
    max_dist = order[:, kk]
    windows = D > max_dist[:, None]                 # True -> outside the neighbourhood -> zeroed
    sigmas = max_dist.copy()
    if s is not None:
        sigmas = np.full(m, float(s))
    sigmas[sigmas == 0] = 1.0

    A = np.exp(-(D ** 2) / sigmas[:, None])         # R recycles column-major => row i / sigmas[i]
    A = A + A.T                                     # symmetrise (diagonal becomes 2)
    A[windows] = 0.0                                # mask by row i's own window, after symmetrising
    rs = A.sum(axis=1)
    rs[rs == 0] = 1.0
    N = A / rs[:, None]
    np.fill_diagonal(N, 1.0)                        # overwritten AFTER normalisation, as in SCOUT
    return N @ X


def log_normalize(counts):
    """SCOUT's normalisation step: plain ``log1p`` of raw counts (no library-size scaling)."""
    return np.log1p(np.asarray(counts, dtype=float))


def scout_preprocess(counts, tree, normalize=True, smoothing_k=None, floor=0.0,
                     scale_height=None, clean_zero_edges=True):
    """Full SCOUT preprocessing. ``counts`` is (n_leaves, n_genes) in tree tip order.

    Returns ``(X, tree)``. Set ``smoothing_k=8`` for their default smoothed arm, or leave it None
    for their log-normalised arm.
    """
    t = clean_edges(tree) if clean_zero_edges else tree
    if scale_height:
        t.scale_height(scale_height)
    X = log_normalize(counts) if normalize else np.asarray(counts, dtype=float)
    if smoothing_k is not None:
        X = lineage_smooth(t, X, smoothing_k)
    if floor and floor > 0:
        X = np.where(X == 0, floor, X)
    return X, t
