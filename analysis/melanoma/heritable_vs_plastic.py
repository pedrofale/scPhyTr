"""Heritable (K) vs plastic (within-subclone) variance decomposition.

Fits the multivariate BM diffusion K *and* the per-gene within-subclone NB
dispersion r jointly by Laplace-EM (``fit_mv_em(..., fit_dispersion=True)``) on
the real melanoma subclone counts. For each gene this yields two variances on the
log-expression scale:

  heritable  V_h = K_gg * T         (BM variance accumulated to the tips, T = root-to-tip time)
  plastic    V_p = trigamma(r_g)    (within-subclone log-variance of the Gamma-Poisson effect)

Their ratio is the count-level analogue of the EVE within/between variance ratio,
and ``V_p / (V_p + V_h)`` is a per-gene plasticity index (the PATH heritability/
plasticity axis): ~0 = heritable/clonal, ~1 = plastic.
"""
import numpy as np
from scipy.special import polygamma

from analysis.melanoma.load import load_counts, tree_leaves, load_tree
from scphytr.inference.laplace import MultiCellPoissonObservation
from scphytr.tools.em import fit_mv_em


def main(n_genes=12):
    X, genes, clone, sf = load_counts()
    # a spread of genes: top-expressed with decent detection
    tot = X.sum(axis=0)
    order = np.argsort(tot)[::-1]
    pick = [g for g in order if (X[:, g] > 0).mean() > 0.6][:n_genes]
    Xg, names = X[:, pick], [genes[g].split("_")[-1] for g in pick]

    leaves = tree_leaves()
    leaf_of = {name: k for k, name in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone])

    tree = load_tree()
    T = float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)

    obs = MultiCellPoissonObservation(Xg, sf, idx, len(leaves), dispersion=10.0)
    res = fit_mv_em(tree, obs, model="BM", fit_dispersion=True, max_em=30, verbose=False)

    K = np.asarray(res.covariance())
    r = res.extra["dispersion"]
    Vh = np.diag(K) * T                 # heritable tip variance (log scale)
    Vp = polygamma(1, r)               # plastic within-subclone log-variance
    frac = Vp / (Vp + Vh)

    print(f"BM + within-subclone dispersion, joint Laplace-EM "
          f"({Xg.shape[0]} cells, {len(leaves)} subclones, {len(pick)} genes, "
          f"{res.extra['em_iters']} EM iters)\n")
    print(f"{'gene':>10} {'rate Kgg':>9} {'V_herit':>8} {'disp r':>8} "
          f"{'V_plast':>8} {'plasticity':>10}")
    o = np.argsort(frac)
    for g in o:
        print(f"{names[g]:>10} {np.diag(K)[g]:9.3f} {Vh[g]:8.2f} {r[g]:8.1f} "
              f"{Vp[g]:8.3f} {frac[g]:10.2f}")
    print(f"\nmost heritable: {names[o[0]]} ({frac[o[0]]:.2f}); "
          f"most plastic: {names[o[-1]]} ({frac[o[-1]]:.2f})")


if __name__ == "__main__":
    main()
