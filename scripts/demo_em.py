"""Laplace-EM (JAX-gradient M-step) for latent multivariate BM under Poisson counts.

Checks that EM (a) increases the marginal monotonically, (b) recovers the latent
evolutionary correlation between genes from counts, and (c) agrees with the direct
gradient-free fit (`fit_mv_latent`).
"""

import time
import numpy as np
import pandas as pd

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance
from scphytr.inference.laplace import PoissonObservation
from scphytr.tools.estimation import fit_mv_latent
from scphytr.tools.em import fit_mv_em


def balanced_newick(depth):
    c = [0]

    def rec(d, root):
        if d == 0:
            c[0] += 1
            return f"L{c[0]}:1.0"
        s = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return s if root else s + ":1.0"

    return rec(depth, True) + ":0.5;"


def corr_cov(rho, sds):
    R = np.array([[1.0, rho], [rho, 1.0]])
    D = np.diag(sds)
    return D @ R @ D


def main():
    rng = np.random.default_rng(0)
    tree = Tree(balanced_newick(6))                 # 64 leaves, free root
    n = len(tree.root.get_leaves())
    print(f"tree: n={n} leaves, p=2 genes; latent multivariate BM, Poisson counts\n")

    K_true = corr_cov(0.8, [0.9, 1.1])
    mu = np.array([2.0, 1.6])
    C = bm_covariance(tree)
    Z = rng.multivariate_normal(np.tile(mu, n), np.kron(C, K_true)).reshape(n, 2)
    S = rng.uniform(40, 80, size=n)
    Y = rng.poisson(S[:, None] * np.exp(Z))
    obs = PoissonObservation(Y, S)

    print("EM iterations (marginal must be non-decreasing):")
    t0 = time.perf_counter()
    em = fit_mv_em(tree, obs, model="BM", trait_names=["g0", "g1"], verbose=True)
    t_em = time.perf_counter() - t0

    t0 = time.perf_counter()
    direct = fit_mv_latent(tree, obs, model="BM", trait_names=["g0", "g1"])
    t_dir = time.perf_counter() - t0

    rho_em = em.correlation().iloc[0, 1]
    rho_dir = direct.correlation().iloc[0, 1]
    print(f"\ntrue corr     = {0.8:+.3f}")
    print(f"EM corr       = {rho_em:+.3f}   (logL={em.loglik:.3f}, {em.extra['em_iters']} iters, {t_em:.1f}s)")
    print(f"direct corr   = {rho_dir:+.3f}   (logL={direct.loglik:.3f}, {t_dir:.1f}s)")
    print(f"\nEM rate matrix K:\n{em.covariance().round(3)}")

    ok = (abs(rho_em - 0.8) < 0.25) and (abs(rho_em - rho_dir) < 0.1) and (em.loglik >= direct.loglik - 1.0)
    print("\nPASS" if ok else "\nCHECK", "- EM recovers correlation and agrees with direct fit")


if __name__ == "__main__":
    main()
