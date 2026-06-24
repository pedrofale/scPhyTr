"""Validate MultiCellPoissonObservation and run an end-to-end count fit.

1. Pure-Poisson branch == PoissonObservation on per-leaf summed counts
   (grad and curvature identical; loglik equal up to an additive constant).
2. NB (within-leaf overdispersion) grad/curvature match finite differences.
3. End-to-end: fit a multivariate BM diffusion K from the real melanoma
   subclone counts via the latent tree-Laplace marginal.
"""
import numpy as np

from scphytr.inference.laplace import (
    PoissonObservation, MultiCellPoissonObservation,
)


def _check_pure_poisson_collapses():
    rng = np.random.default_rng(0)
    n_leaves, p = 5, 3
    # random cell -> leaf assignment, 4..9 cells per leaf
    idx = np.concatenate([[l] * rng.integers(4, 10) for l in range(n_leaves)])
    n_cells = idx.shape[0]
    y = rng.poisson(5.0, size=(n_cells, p)).astype(float)
    S = rng.uniform(0.5, 2.0, size=n_cells)
    mc = MultiCellPoissonObservation(y, S, idx, n_leaves)         # dispersion=None
    # summed-count reference
    Ytot = np.zeros((n_leaves, p)); Stot = np.zeros((n_leaves, p))
    np.add.at(Ytot, idx, y); np.add.at(Stot, idx, S[:, None])
    ref = PoissonObservation(Ytot, Stot)
    F = rng.standard_normal((n_leaves, p)) * 0.3
    assert np.allclose(mc.grad(F), ref.grad(F)), "grad mismatch"
    assert np.allclose(mc.neg_hess_diag(F), ref.neg_hess_diag(F)), "curvature mismatch"
    # loglik equal up to an F-independent constant
    d1 = mc.loglik(F) - ref.loglik(F)
    d2 = mc.loglik(F + 0.7) - ref.loglik(F + 0.7)
    assert abs(d1 - d2) < 1e-8, "loglik differs by more than a constant"
    print("[1] pure-Poisson multi-cell == summed-count PoissonObservation  OK")


def _check_nb_finite_diff():
    rng = np.random.default_rng(1)
    n_leaves, p = 3, 2
    idx = np.concatenate([[l] * rng.integers(3, 7) for l in range(n_leaves)])
    y = rng.poisson(4.0, size=(idx.shape[0], p)).astype(float)
    S = rng.uniform(0.7, 1.5, size=idx.shape[0])
    r = np.array([2.0, 8.0])
    mc = MultiCellPoissonObservation(y, S, idx, n_leaves, dispersion=r)
    F = rng.standard_normal((n_leaves, p)) * 0.3
    g = mc.grad(F)
    h = mc.neg_hess_diag(F)
    # second-difference roundoff ~ machine_eps/step^2, so use a larger step here
    eps = 1e-4
    gfd = np.zeros_like(F); hfd = np.zeros_like(F)
    l0 = mc.loglik(F)
    for i in range(n_leaves):
        for j in range(p):
            Fp = F.copy(); Fp[i, j] += eps
            Fm = F.copy(); Fm[i, j] -= eps
            gfd[i, j] = (mc.loglik(Fp) - mc.loglik(Fm)) / (2 * eps)
            hfd[i, j] = -(mc.loglik(Fp) - 2 * l0 + mc.loglik(Fm)) / eps ** 2
    assert np.allclose(g, gfd, atol=1e-5), f"NB grad vs FD\n{g}\n{gfd}"
    assert np.allclose(h, hfd, atol=1e-2, rtol=1e-3), f"NB curvature vs FD\n{h}\n{hfd}"
    # r -> inf recovers pure Poisson grad/curvature
    mc_big = MultiCellPoissonObservation(y, S, idx, n_leaves, dispersion=1e9)
    mc_pois = MultiCellPoissonObservation(y, S, idx, n_leaves)
    assert np.allclose(mc_big.grad(F), mc_pois.grad(F), atol=1e-3)
    print("[2] NB grad/curvature match finite differences; r->inf -> Poisson  OK")


def _run_real_data(n_genes=5):
    from analysis.melanoma.load import load_counts, tree_leaves, load_tree
    from scphytr.tools.em import fit_mv_em
    from scphytr.tools.estimation import cov_to_corr

    leaves = tree_leaves()
    X, genes, clone, sf = load_counts()
    # most-expressed, non-trivial genes as a quick demo set
    tot = X.sum(axis=0)
    pick = np.argsort(tot)[::-1]
    pick = [g for g in pick if (X[:, g] > 0).mean() > 0.5][:n_genes]
    Xg = X[:, pick]
    leaf_of = {name: k for k, name in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone])

    tree = load_tree()                          # floors zero-length branches
    obs = MultiCellPoissonObservation(Xg, sf, idx, len(leaves), dispersion=10.0)
    res = fit_mv_em(tree, obs, model="BM", max_em=20)
    K = np.asarray(res.covariance())
    print(f"[3] fit BM K from real subclone counts ({Xg.shape[0]} cells, "
          f"{len(pick)} genes, NB within-clone overdispersion):")
    print("    per-gene evolutionary rates (diag K):", np.round(np.diag(K), 3))
    R = cov_to_corr(K)
    iu = np.triu_indices(len(pick), 1)
    print(f"    evolutionary correlations: mean|rho|={np.abs(R[iu]).mean():.2f}, "
          f"range [{R[iu].min():.2f}, {R[iu].max():.2f}]")


if __name__ == "__main__":
    _check_pure_poisson_collapses()
    _check_nb_finite_diff()
    _run_real_data()
