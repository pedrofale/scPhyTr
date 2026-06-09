"""Estimate latent evolutionary correlations between genes under a Poisson model.

The genes' continuous latent log-expression evolves by a *multivariate* BM with a
full diffusion matrix K (their evolutionary rates are correlated). We only observe
Poisson counts at the leaves, yet by maximizing the multivariate latent
tree-Laplace marginal we recover K's correlation structure.

Also checks that, in the Gaussian (directly-observed) limit, the latent fitter
agrees with the closed-form contrast MLE.
"""

import numpy as np
import pandas as pd

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance
from scphytr.inference.laplace import PoissonObservation, GaussianObservation
from scphytr.tools.estimation import (
    fit_bm_mv, fit_mv_latent, estimate_correlation, cov_to_corr,
)


def balanced_newick(depth):
    c = [0]

    def rec(d, root):
        if d == 0:
            c[0] += 1
            return f"L{c[0]}:1.0"
        s = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return s if root else s + ":1.0"

    return rec(depth, True) + ":0.01;"


def corr_matrix(rho, sds):
    R = np.array([[1.0, rho], [rho, 1.0]])
    D = np.diag(sds)
    return D @ R @ D


def simulate_latent(tree, K, mu, rng):
    n = len(tree.root.get_leaves())
    p = K.shape[0]
    C = bm_covariance(tree)
    Z = rng.multivariate_normal(np.tile(mu, n), np.kron(C, K)).reshape(n, p)
    return Z


def main():
    rng = np.random.default_rng(0)
    tree = Tree(balanced_newick(6))                  # 64 leaves
    n = len(tree.root.get_leaves())
    leaf_names = tree.phylotree.get_leaf_names()

    print(f"tree: n={n} leaves, p=2 genes; latent multivariate BM, Poisson counts\n")

    # ---- Gaussian-limit sanity: latent fitter vs closed-form contrast MLE ----
    K_true = corr_matrix(0.7, [0.9, 1.2])
    mu = np.array([2.0, 1.5])
    Z = simulate_latent(tree, K_true, mu, rng)
    trait_table = pd.DataFrame(Z, index=leaf_names, columns=["g0", "g1"])

    closed = fit_bm_mv(tree, trait_table)
    latent_gauss = fit_mv_latent(tree, GaussianObservation(Z, 1e-3), model="BM",
                                 trait_names=["g0", "g1"])
    r_closed = closed.correlation().iloc[0, 1]
    r_latent = latent_gauss.correlation().iloc[0, 1]
    print("Gaussian limit (should match):")
    print(f"  contrast-MLE corr = {r_closed:+.3f} | latent-fit corr = {r_latent:+.3f} "
          f"| |diff| = {abs(r_closed - r_latent):.3f}\n")

    # ---- Poisson counts over the correlated latent ----
    print("Poisson counts, recover latent evolutionary correlation:")
    results = []
    for rho_true in (0.8, 0.0, -0.6):
        K = corr_matrix(rho_true, [0.9, 1.1])
        Z = simulate_latent(tree, K, mu, rng)
        S = rng.uniform(40, 80, size=n)
        Y = rng.poisson(S[:, None] * np.exp(Z))
        obs = PoissonObservation(Y, S)
        fit = fit_mv_latent(tree, obs, model="BM", trait_names=["g0", "g1"])
        rho_hat = fit.correlation().iloc[0, 1]
        results.append((rho_true, rho_hat))
        print(f"  rho_true = {rho_true:+.2f}  ->  rho_hat = {rho_hat:+.2f}")

    err = np.mean([abs(a - b) for a, b in results])
    print(f"\nmean |rho_true - rho_hat| = {err:.3f}")
    if err < 0.25:
        print("PASS: latent evolutionary correlation recovered from Poisson counts")
    else:
        print("CHECK: recovery error larger than expected (small tree / MC noise)")


if __name__ == "__main__":
    main()
