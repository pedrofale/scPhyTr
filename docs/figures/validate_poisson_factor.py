"""Validation for Poisson phylogenetic factor analysis.

Simulates raw counts from the low-rank Poisson model
``y_ig ~ Poisson(s_i exp(mu_g + (W z_i)_g))`` with factor scores ``z`` evolving by
Brownian motion on a tree, then checks that ``fit_poisson_factor_analysis``
(maximum likelihood via Laplace-EM) recovers the gene programs. As for the
Gaussian PFA, the loading *subspace* is invariant to the cell (row) covariance,
so the phylogenetic fit and a phylogeny-naive (star-tree) fit recover the same
subspace -- the tree's effect is on the factor trajectories and on inference
about their evolutionary dynamics, not on the static program subspace.

Run: ``python -m docs.figures.validate_poisson_factor`` (or ``python`` this file
from the repo root).
"""
import warnings; warnings.filterwarnings("ignore")
import random

import numpy as np
import ete3

from scphytr.utils.tree import Tree
from scphytr.tools.poisson_factor import (
    fit_poisson_factor_analysis, simulate_poisson_pfa)
from scphytr.tools.factor_analysis import subspace_error, principal_angles


def wrap(et):
    """Wrap an ete3 tree in the scphytr Tree container (no covariance build)."""
    T = Tree()
    T.phylotree = et
    T.root = et.get_tree_root()
    return T


def random_tree(n, seed=0):
    """A random bifurcating tree, normalized to unit root-to-tip height."""
    rng = np.random.default_rng(seed)
    random.seed(seed)                    # ete3.populate uses the `random` module
    t = ete3.Tree()
    t.populate(n)
    for i, l in enumerate(t.get_leaves()):
        l.name = f"c{i}"
    for nd in t.traverse():
        if not nd.is_root():
            nd.dist = float(rng.uniform(0.2, 1.0))
    t.get_tree_root().dist = 1e-6
    h = max(t.get_distance(l) for l in t.get_leaves())
    for nd in t.traverse():
        nd.dist = nd.dist / h
    t.get_tree_root().dist = 1.0
    return t


def star_tree(leaf_names):
    """A star tree (all leaves off the root): the phylogeny-naive baseline."""
    t = ete3.Tree()
    t.get_tree_root().dist = 1.0
    for nm in leaf_names:
        t.add_child(name=nm, dist=1.0)
    return t


def eta_corr(fit, Xtrue, Wtrue):
    """Rotation-invariant recovery of the low-dim log-rate signal eta = X W^T."""
    return np.corrcoef((fit.scores @ fit.W.T).ravel(),
                       (Xtrue @ Wtrue.T).ravel())[0, 1]


def _corr(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-300, None))
    return K / np.outer(d, d)


def corrmat_recovery(fit, Wtrue):
    """Correlation between estimated and true gene-gene evolutionary corr (K=WW^T).

    Compares off-diagonal entries of ``corr(W_fit W_fit^T)`` with the truth -- the
    deconfounded gene-gene correlation matrix the factor model targets.
    """
    Rhat = fit.evolutionary_correlation()
    Rtrue = _corr(Wtrue @ Wtrue.T)
    iu = np.triu_indices_from(Rhat, k=1)
    return np.corrcoef(Rhat[iu], Rtrue[iu])[0, 1]


def one_replicate(seed, n=300, p=45, k=3):
    rng = np.random.default_rng(1000 + seed)
    tree = wrap(random_tree(n, seed=seed))

    # block-structured loadings with clearly distinct per-factor variance
    W = np.zeros((p, k))
    scales = [1.2, 0.9, 0.6]
    for j in range(k):
        W[j * (p // k):(j + 1) * (p // k), j] = scales[j] * rng.uniform(0.7, 1.0, p // k) * (1 if j % 2 else -1)
    W += 0.02 * rng.standard_normal((p, k))
    mu = rng.uniform(-5.0, -3.0, p)

    Y, Xtrue, _, names = simulate_poisson_pfa(tree, W, mu, mean_size=3000.0, seed=seed)

    fit = fit_poisson_factor_analysis(Y, tree, k=k, leaf_names=names, n_iter=40)
    fit_n = fit_poisson_factor_analysis(Y, wrap(star_tree(names)), k=k,
                                        leaf_names=names, n_iter=40)
    return {
        "zeros": float(np.mean(Y == 0)), "umi": float(np.median(Y.sum(1))),
        "monotone": bool(np.all(np.diff(fit.history) > -1e-2)),
        "se_phylo": subspace_error(fit.W, W), "se_naive": subspace_error(fit_n.W, W),
        "eta_phylo": eta_corr(fit, Xtrue, W), "eta_naive": eta_corr(fit_n, Xtrue, W),
        "Rcorr_phylo": corrmat_recovery(fit, W), "Rcorr_naive": corrmat_recovery(fit_n, W),
        "ang_phylo": float(np.degrees(principal_angles(fit.W, W)).max()),
    }


def main():
    n, p, k = 300, 45, 3
    print(f"Poisson phylogenetic factor analysis -- validation (n={n}, p={p}, k={k})\n")
    rows = []
    for seed in range(5):
        r = one_replicate(seed, n, p, k)
        rows.append(r)
        print(f"seed {seed}: UMI~{r['umi']:.0f} zeros={r['zeros']:.2f} mono={r['monotone']} | "
              f"subspace_err phylo={r['se_phylo']:.3f}/naive={r['se_naive']:.3f} | "
              f"gene-gene corr(K) recovery phylo={r['Rcorr_phylo']:.3f}/naive={r['Rcorr_naive']:.3f}")

    def avg(key):
        return float(np.mean([r[key] for r in rows]))
    print("\nAVERAGE over 5 replicates")
    print(f"  EM monotone in all       : {all(r['monotone'] for r in rows)}")
    print(f"  loading subspace error   : phylo {avg('se_phylo'):.3f} | naive {avg('se_naive'):.3f}  (max {np.sqrt(k):.2f})")
    print(f"  low-dim signal recovery  : phylo {avg('eta_phylo'):.3f} | naive {avg('eta_naive'):.3f}")
    print(f"  gene-gene corr(K=WW^T)   : phylo {avg('Rcorr_phylo'):.3f} | naive {avg('Rcorr_naive'):.3f}")


if __name__ == "__main__":
    main()
