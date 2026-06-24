"""Transcriptome-scale per-gene adaptive scan via the batched engine.

Fits BM / OU-1 / OU-2 for all genes at once (``scphytr.inference.batched``) with
a per-cell NB leaf observation (within-subclone overdispersion), AIC-selects, and
writes a per-gene table. An OU-2 win = adaptively shifted expression in the
chosen sublines. Validates the batched marginal against the per-gene path.
"""
import time
import numpy as np
import pandas as pd

from analysis.melanoma.load import load_counts, tree_leaves, load_tree, load_regimes
from scphytr.inference.batched import BatchedTreeLaplace, PoissonLeaf, NBLeaf


def _grids(tree):
    H = float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)
    amax = 30.0 / H
    alpha_grid = np.r_[0.0, np.geomspace(0.02 * amax, amax, 7)]
    sigma2_grid = np.geomspace(1e-3, 1.0, 16)
    return alpha_grid, sigma2_grid


def fit_adaptive(tree, X, idx, nL, sf, cols, regimes, n_reg):
    """BM/OU1/OU2 batched fit (per-cell NB) + AIC selection -> per-gene DataFrame."""
    alpha_grid, sigma2_grid = _grids(tree)
    Xc = X[:, cols]
    r = NBLeaf.estimate_r(Xc, sf, idx, nL)
    obs = NBLeaf(Xc, sf, idx, nL, r)
    obs_r = lambda regs=None: obs   # same obs; regime only changes the tree model
    bm = BatchedTreeLaplace(tree).fit(obs, [0.0], sigma2_grid)
    ou1 = BatchedTreeLaplace(tree).fit(obs, alpha_grid, sigma2_grid)
    ou2 = BatchedTreeLaplace(tree, regimes=regimes).fit(obs, alpha_grid, sigma2_grid)
    aic = {"BM": 4 - 2 * bm["logml"], "OU1": 6 - 2 * ou1["logml"],
           "OU2": 2 * (n_reg + 2) - 2 * ou2["logml"]}
    A = np.vstack([aic["BM"], aic["OU1"], aic["OU2"]])
    sel_i = A.argmin(0)
    names = np.array(["BM", "OU1", "OU2"])
    return pd.DataFrame({
        "selected": names[sel_i], "adaptive": sel_i == 2,
        "aic_BM": aic["BM"], "aic_OU1": aic["OU1"], "aic_OU2": aic["OU2"],
        "disp_r": r, "alpha_OU1": ou1["alpha"],
        "dtheta_OU2": np.abs(ou2["theta"][1] - ou2["theta"][0]),
    })


def _load(regime="har"):
    X, genes, clone, sf = load_counts()
    leaves = tree_leaves(); leaf_of = {n: k for k, n in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone]); nL = len(leaves)
    tree = load_tree()
    regimes, n_reg = load_regimes(tree, regime)
    return X, genes, idx, nL, sf, tree, regimes, n_reg


def _validate(tree, X, idx, nL, sf):
    """Batched marginal == per-gene scalar path (same per-cell NB obs)."""
    from scphytr.inference.laplace import MultiCellPoissonObservation
    from scphytr.inference.tree_laplace import latent_tree_laplace_marginal
    cols = [int(g) for g in np.argsort(X.sum(0))[::-1][:5]]
    Xc = X[:, cols]
    r = NBLeaf.estimate_r(Xc, sf, idx, nL)
    obs = NBLeaf(Xc, sf, idx, nL, r)
    B = BatchedTreeLaplace(tree)
    theta = np.tile(obs.init_leaf().mean(0), (1, 1))
    sig = np.full(len(cols), 0.05)
    worst = 0.0
    for al in (0.0, 0.5):
        ml, th = B.marginal(obs, al, sig, theta, n_inner=0)
        for j, g in enumerate(cols):
            o1 = MultiCellPoissonObservation(Xc[:, [j]], sf, idx, nL,
                                             dispersion=float(r[j]), univariate=True)
            ref = latent_tree_laplace_marginal(tree, o1, al, float(th[0, j]),
                                               float(sig[j]), root_value=float(th[0, j]))
            worst = max(worst, abs(ml[j] - ref))
    print(f"batched NB marginal vs per-gene scalar path: worst |diff| = {worst:.2e}")


def main():
    X, genes, idx, nL, sf, tree, regimes, n_reg = _load("har")
    _validate(tree, X, idx, nL, sf)
    print("\nbatched BM/OU1/OU2 adaptive fit (per-cell NB), scaling:")
    for G in (200, 2000):
        cols = [int(g) for g in np.argsort(X.sum(0))[::-1][:G]]
        t = time.time()
        df = fit_adaptive(tree, X, idx, nL, sf, cols, regimes, n_reg)
        dt = time.time() - t
        print(f"  G={G:5d}: {dt:6.1f} s ({dt/G*1000:.2f} ms/gene)  "
              f"median r={np.median(df['disp_r']):.1f}  "
              f"adaptive={int(df['adaptive'].sum())} ({df['adaptive'].mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
