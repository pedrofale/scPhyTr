"""Shared helpers for phylo-aware vs naive factor analysis on a fixed tree.

These mirror the validated logic in ``docs/figures/make_felsenstein_figure.py``
but take a phylogenetic covariance ``C`` directly, so they apply to a real tree.
"""

import numpy as np


def chol(C, jitter=1e-8):
    return np.linalg.cholesky(C + jitter * np.eye(C.shape[0]))


def simulate_independent_genes(L_C, p, rng):
    """Each gene an *independent* Brownian motion on the tree (K diagonal).

    No shared factor exists, so any factor a method finds is a phylogenetic
    artifact. Returns Y (n, p).
    """
    return L_C @ rng.standard_normal((L_C.shape[0], p))


def phylo_contrasts(Y, C):
    """Felsenstein's n-1 standardized independent contrasts (multivariate).

    Orthonormal contrast basis H (H^T 1 = 0) removes the unknown root/mean;
    whitening by the Cholesky of H^T C H yields i.i.d. rows for an independent-BM
    gene. Genuinely exchangeable, so the permutation null below is valid.
    """
    n = C.shape[0]
    M = np.eye(n) - np.ones((n, n)) / n
    Uc, _, _ = np.linalg.svd(M)
    H = Uc[:, : n - 1]
    G = np.linalg.cholesky(H.T @ C @ H)
    return np.linalg.solve(G, H.T @ Y)


def parallel_analysis(M, n_perm, rng, n_eigs=None):
    """Horn's parallel analysis: # leading factors beating the i.i.d. null.

    Valid only if the rows of ``M`` are exchangeable (true for contrasts, false
    for raw phylogenetic leaves). Counts only *leading consecutive* components
    that exceed the 95th-percentile null, avoiding multiple-comparison inflation.
    """
    m, p = M.shape
    n_eigs = p if n_eigs is None else min(n_eigs, p)
    Mc = M - M.mean(0)
    obs = np.sort(np.linalg.eigvalsh(np.cov(Mc, rowvar=False)))[::-1]
    null = np.empty((n_perm, p))
    for b in range(n_perm):
        P = np.empty_like(Mc)
        for g in range(p):
            P[:, g] = Mc[rng.permutation(m), g]
        null[b] = np.sort(np.linalg.eigvalsh(np.cov(P, rowvar=False)))[::-1]
    null95 = np.percentile(null, 95, axis=0)
    n_sig = 0
    for i in range(n_eigs):
        if obs[i] > null95[i]:
            n_sig += 1
        else:
            break
    return obs[:n_eigs], null95[:n_eigs], n_sig


def get_clades(tree, leaf_names, Q):
    """Partition leaves into ``Q`` monophyletic clades (labels in leaf order).

    Repeatedly split the clade with the most leaves into its children until
    there are Q clades; used to measure how much of a program's variance lives
    *between* deep clones (phylogenetic signal) vs within them.
    """
    root = tree.get_tree_root()
    clades = [root]
    while len(clades) < Q:
        # split the largest clade that still has children
        splittable = [c for c in clades if not c.is_leaf()]
        if not splittable:
            break
        node = max(splittable, key=lambda c: len(c.get_leaves()))
        clades.remove(node)
        clades.extend(node.children)
    pos = {nm: i for i, nm in enumerate(leaf_names)}
    labels = np.full(len(leaf_names), -1)
    for cl_id, node in enumerate(clades):
        for leaf in node.get_leaves():
            if leaf.name in pos:
                labels[pos[leaf.name]] = cl_id
    return labels


def clade_eta2(t, labels):
    """Fraction of a score vector's variance lying *between* clades (eta^2).

    High eta^2 => the program mostly distinguishes deep clones (clonal identity /
    phylogenetic axis); low eta^2 => mostly within-clade (co-regulation).
    """
    t = np.asarray(t, float)
    grand = t.mean()
    ss_tot = np.sum((t - grand) ** 2)
    ss_between = 0.0
    for c in np.unique(labels):
        ti = t[labels == c]
        ss_between += len(ti) * (ti.mean() - grand) ** 2
    return float(ss_between / (ss_tot + 1e-12))


def deep_clade_indicator(tree, leaf_names):
    """A 0/1 label splitting leaves by the largest balanced split near the root.

    Walks down from the root taking the largest child until both sides are
    sizeable, giving the dominant deep bipartition (the tumor's main subclonal
    division). Returned in ``leaf_names`` order.
    """
    node = tree.get_tree_root()
    while len(node.children) == 1:
        node = node.children[0]
    # pick the split whose two largest children are most balanced
    best = None
    cur = node
    for _ in range(50):
        kids = sorted(cur.children, key=lambda c: len(c.get_leaves()), reverse=True)
        if len(kids) < 2:
            cur = kids[0] if kids else cur
            continue
        big = kids[0]
        side = set(l.name for l in big.get_leaves())
        bal = min(len(side), len(leaf_names) - len(side)) / len(leaf_names)
        if best is None or bal > best[1]:
            best = (side, bal)
        if bal > 0.3:
            break
        cur = big
    side = best[0]
    return np.array([1 if nm in side else 0 for nm in leaf_names])
