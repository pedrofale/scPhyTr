"""Validate the multivariate latent tree-Laplace marginal.

Oracles:
  * p=1 must reproduce the univariate latent tree-Laplace exactly.
  * Gaussian (identity-with-noise) observations make Laplace exact, so the
    marginal must equal the closed-form MVN with covariance C (x) K + noise.
Covers BM and OU-1 over a few trees and trait dimensions.
"""

import numpy as np
from scipy.stats import multivariate_normal

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance, ou_covariance
from scphytr.inference.laplace import PoissonObservation, GaussianObservation
from scphytr.inference.tree_laplace import latent_tree_laplace_marginal
from scphytr.inference.tree_laplace_mv import mv_tree_laplace_marginal


def balanced_newick(depth):
    c = [0]

    def rec(d, root):
        if d == 0:
            c[0] += 1
            return f"L{c[0]}:1.0"
        s = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return s if root else s + ":1.0"

    return rec(depth, True) + ":0.01;"


TREES = {
    "8tip": "(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):0.01;",
    "balanced-32": balanced_newick(5),
}


def random_spd(p, rng, scale=1.0):
    A = rng.standard_normal((p, p))
    return scale * (A @ A.T / p + np.eye(p))


def check_p1_reduction(seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for nwk in TREES.values():
        tree = Tree(nwk)
        n = len(tree.root.get_leaves())
        for alpha, theta in [(0.0, 1.5), (1.3, 2.0)]:
            s2 = float(rng.uniform(0.4, 1.2))
            S = rng.uniform(15, 35, size=n)
            Yvec = rng.poisson(S * np.exp(theta + np.sqrt(s2) * rng.standard_normal(n)))
            uni = latent_tree_laplace_marginal(
                tree, PoissonObservation(Yvec, S), alpha, theta, s2, root_value=theta)
            mv = mv_tree_laplace_marginal(
                tree, PoissonObservation(Yvec[:, None], S), alpha,
                np.array([theta]), np.array([[s2]]), root_value=np.array([theta]))
            worst = max(worst, abs(uni - mv))
    print(f"p=1 reduction (mv vs univariate): worst |diff| = {worst:.3e}")
    return worst


def check_gaussian_exact(seed=1):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for name, nwk in TREES.items():
        tree = Tree(nwk)
        n = len(tree.root.get_leaves())
        for p in (2, 3):
            for kind in ("BM", "OU"):
                alpha = 0.0 if kind == "BM" else float(rng.uniform(0.4, 1.6))
                theta = rng.uniform(1.0, 3.0, size=p)
                K = random_spd(p, rng, scale=0.8)
                C = bm_covariance(tree) if kind == "BM" else ou_covariance(tree, alpha)

                Sigma = np.kron(C, K)                      # leaf-major (leaf outer, gene inner)
                mean = np.tile(theta, n)
                tau = rng.uniform(0.05, 0.3, size=(n, p))
                z = rng.multivariate_normal(mean, Sigma).reshape(n, p)
                Y = z + np.sqrt(tau) * rng.standard_normal((n, p))

                exact = multivariate_normal.logpdf(
                    Y.ravel(), mean, Sigma + np.diag(tau.ravel()))
                mv = mv_tree_laplace_marginal(
                    tree, GaussianObservation(Y, tau), alpha, theta, K,
                    root_value=theta)
                worst = max(worst, abs(exact - mv))
                print(f"  [{name} p={p} {kind}] |exact - mv| = {abs(exact - mv):.3e}")
    print(f"Gaussian exact: worst |diff| = {worst:.3e}")
    return worst


def main():
    print("Multivariate latent tree-Laplace validation")
    w1 = check_p1_reduction()
    print()
    w2 = check_gaussian_exact()
    print()
    tol = 1e-6
    worst = max(w1, w2)
    if worst < tol:
        print(f"PASS: multivariate marginal matches oracles (worst {worst:.3e} < {tol:g})")
    else:
        print(f"FAIL: worst {worst:.3e} exceeds {tol:g}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
