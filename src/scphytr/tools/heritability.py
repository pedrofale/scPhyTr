"""Phylogenetic signal / heritability for a continuous trait (Pagel's lambda).

PATH measures heritability as phylogenetic autocorrelation (Moran's I). The
classical model-based analogue is **Pagel's lambda**: fit the trait as Brownian
motion on the tree but scale the *off-diagonal* (shared-ancestry) entries of the
tip covariance by lambda in [0, 1]. lambda = 1 is the full tree (maximally
heritable); lambda = 0 is a star (tips i.i.d., no phylogenetic signal / fully
plastic). lambda_hat is a bounded, interpretable heritability statistic, and a
likelihood-ratio test against lambda = 0 gives significance.

Unlike the OU mean-reversion alpha (which saturates for weak signal), lambda is a
direct proportion-of-variance read and does not saturate -- so it is the right
scPhyTr counterpart to Moran's I. This is the dense O(n^3) estimator, exact and
intended for trees / subsamples up to a few thousand tips.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def shared_ancestry_cov(tree):
    """Brownian tip covariance C: C_ij = root-to-MRCA(i,j) time, C_ii = root-to-tip."""
    leaves = tree.phylotree.get_leaves()
    n = len(leaves)
    idx = {id(l): k for k, l in enumerate(leaves)}
    depth = {}
    for nd in tree.phylotree.traverse("preorder"):
        depth[id(nd)] = 0.0 if nd.up is None else depth[id(nd.up)] + nd.dist
    C = np.zeros((n, n))
    for l in leaves:
        C[idx[id(l)], idx[id(l)]] = depth[id(l)]
    for nd in tree.phylotree.traverse("postorder"):
        ch = nd.children
        if len(ch) < 2:
            continue
        d = depth[id(nd)]
        groups = [[idx[id(x)] for x in c.get_leaves()] for c in ch]
        for i in range(len(groups)):
            gi = np.array(groups[i])
            for j in range(i + 1, len(groups)):
                gj = np.array(groups[j])
                C[np.ix_(gi, gj)] = d
                C[np.ix_(gj, gi)] = d
    return C, [l.name for l in leaves]


def _profile_ll(lam, C, di, y, ones, n):
    """Profiled BM-with-lambda log-likelihood (GLS mean, profiled sigma^2)."""
    Cl = lam * C
    np.fill_diagonal(Cl, di)
    Cl[np.diag_indices(n)] += 1e-8 * di.mean()
    try:
        cf = cho_factor(Cl, lower=True)
    except np.linalg.LinAlgError:
        return -np.inf
    Cinv_y = cho_solve(cf, y)
    Cinv_1 = cho_solve(cf, ones)
    mu = float(ones @ Cinv_y) / float(ones @ Cinv_1)
    r = y - mu
    q = float(r @ cho_solve(cf, r))
    sigma2 = q / n
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return -0.5 * (n * np.log(2 * np.pi * sigma2) + logdet + n)


def pagels_lambda(tree, values, C=None):
    """Estimate Pagel's lambda heritability + LR test against lambda=0 (star).

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    values : dict {leaf_name -> trait value}
    C : optional precomputed (C, names) from :func:`shared_ancestry_cov`.

    Returns
    -------
    dict with ``lambda`` (heritability in [0,1]), ``loglik``, ``loglik0``
    (lambda=0), ``lr`` (2*Delta), and ``p`` (chi2_1 LR p-value vs no signal).
    """
    if C is None:
        C, names = shared_ancestry_cov(tree)
    else:
        C, names = C
    y = np.array([float(values[nm]) for nm in names])
    n = len(y)
    di = np.diag(C).copy()
    ones = np.ones(n)

    res = minimize_scalar(lambda lam: -_profile_ll(lam, C, di, y, ones, n),
                          bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": 1e-4})
    lam = float(res.x)
    ll = -res.fun
    ll0 = _profile_ll(0.0, C, di, y, ones, n)
    lr = max(2.0 * (ll - ll0), 0.0)
    p = 0.5 * chi2.sf(lr, 1)          # boundary (lambda>=0): half chi2_1
    return {"lambda": lam, "loglik": ll, "loglik0": ll0, "lr": lr, "p": p}
