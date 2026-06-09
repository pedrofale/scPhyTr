"""Validate the Laplace-approximate Poisson marginal likelihood.

On a small tree we compare ``laplace_marginal_loglik`` for a Poisson observation
model with a BM latent prior against a brute-force numerical integral over the
latent values. Laplace is approximate, so we expect closeness (small absolute
error in log-marginal), not machine precision.
"""

import numpy as np
import ete3

import scphytr
from scphytr.utils.covariance import bm_covariance, ou_covariance, ou_regime_mean
from scphytr.utils.pruning import paint_regimes
from scphytr.inference.laplace import (
    PoissonObservation, GaussianObservation, laplace_marginal_loglik, laplace_posterior,
)
from scphytr.inference.tree_laplace import latent_tree_laplace_marginal
from scipy.stats import multivariate_normal
from scipy.special import gammaln


def build_tree(nwk):
    tree = scphytr.utils.Tree()
    tree.phylotree = ete3.PhyloTree(nwk, format=1)
    tree.root = tree.phylotree.get_tree_root()
    return tree


def brute_force_logmarginal(obs, mean, Sigma, grid=81, span=8.0):
    """log ∫ p(Y|z) N(z; mean, Sigma) dz by tensor-grid integration.

    The grid is centered on the Laplace posterior mode and scaled by the posterior
    standard deviations, so it resolves the (possibly sharp) posterior peak even
    when counts are large.
    """
    n = mean.shape[0]
    post = laplace_posterior(obs, mean, Sigma)
    center = post["mode"]
    sd = np.sqrt(np.clip(np.diag(post["cov"]), 1e-12, None))
    axes = [np.linspace(center[i] - span * sd[i], center[i] + span * sd[i], grid) for i in range(n)]
    dz = [ax[1] - ax[0] for ax in axes]

    Sigma_inv = np.linalg.inv(Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)
    const = -0.5 * (n * np.log(2 * np.pi) + logdet)

    mesh = np.meshgrid(*axes, indexing="ij")
    Z = np.stack([m.ravel() for m in mesh], axis=1)  # (grid^n, n)

    d = Z - mean
    quad = np.einsum("ki,ij,kj->k", d, Sigma_inv, d)
    log_prior = const - 0.5 * quad

    rate = obs.S[None, :] * np.exp(Z)
    log_lik = np.sum(obs.y[None, :] * Z - rate - gammaln(obs.y + 1.0)[None, :], axis=1)

    log_integrand = (log_prior + log_lik).reshape([grid] * n)
    # log of trapezoid sum via logsumexp with cell-volume weight.
    from scipy.special import logsumexp
    log_vol = np.sum(np.log(dz))
    return logsumexp(log_integrand) + log_vol


def balanced_newick(depth):
    counter = [0]

    def rec(d, is_root):
        if d == 0:
            counter[0] += 1
            return f"L{counter[0]}:1.0"
        node = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return node if is_root else node + ":1.0"

    return rec(depth, True) + ";"


def _prior_mean_cov(tree, kind, alpha, theta, sigma2, regimes, root_value):
    """Leaf prior mean and covariance for a given model (for dense oracles)."""
    n = len(tree.root.get_leaves())
    if kind == "BM":
        return theta * np.ones(n), sigma2 * bm_covariance(tree)
    if kind == "OU":
        return theta * np.ones(n), sigma2 * ou_covariance(tree, alpha)
    if kind == "OU2":
        mean = ou_regime_mean(tree, alpha, theta, regimes, root_value=root_value)
        return mean, sigma2 * ou_covariance(tree, alpha)
    raise ValueError(kind)


def check_tree_laplace(name, nwk, n_trials=4, seed=0):
    """tree-Laplace == exact (Gaussian obs) and == dense-Laplace (Poisson),
    across BM, OU-1, and OU-2 models."""
    rng = np.random.default_rng(seed)
    tree = build_tree(nwk)
    n = len(tree.root.get_leaves())

    shift_node = tree.root.children[0]
    regimes, n_regimes = paint_regimes(tree, [shift_node])

    worst_gauss = 0.0
    worst_pois = 0.0
    for _ in range(n_trials):
        alpha = float(rng.uniform(0.3, 1.8))
        sigma2 = float(rng.uniform(0.3, 1.0))
        S = rng.uniform(10.0, 40.0, size=n)

        for kind in ("BM", "OU", "OU2"):
            if kind == "BM":
                a_eff, theta = 0.0, float(rng.uniform(1.5, 3.0))
                rv = theta
            elif kind == "OU":
                a_eff, theta = alpha, float(rng.uniform(1.5, 3.0))
                rv = theta
            else:
                a_eff = alpha
                theta = rng.uniform(1.5, 3.0, size=n_regimes)
                rv = theta[regimes[tree.root]]

            mean, Sigma = _prior_mean_cov(tree, kind, a_eff, theta, sigma2, regimes, rv)

            # Gaussian observations: Laplace is exact -> tree must equal closed form.
            tau = rng.uniform(0.05, 0.4, size=n)
            y = rng.multivariate_normal(mean, Sigma) + np.sqrt(tau) * rng.standard_normal(n)
            gobs = GaussianObservation(y, tau)
            exact = multivariate_normal.logpdf(y, mean, Sigma + np.diag(tau))
            tree_g = latent_tree_laplace_marginal(tree, gobs, a_eff, theta, sigma2,
                                                  regimes=(regimes if kind == "OU2" else None),
                                                  root_value=rv)
            worst_gauss = max(worst_gauss, abs(exact - tree_g))

            # Poisson observations: tree-Laplace must equal dense-Laplace.
            z = rng.multivariate_normal(mean, Sigma)
            Y = rng.poisson(S * np.exp(z))
            pobs = PoissonObservation(Y, S)
            dense = laplace_marginal_loglik(pobs, mean, Sigma)
            tree_p = latent_tree_laplace_marginal(tree, pobs, a_eff, theta, sigma2,
                                                  regimes=(regimes if kind == "OU2" else None),
                                                  root_value=rv)
            worst_pois = max(worst_pois, abs(dense - tree_p))

    print(f"[{name}] n={n}: tree vs exact (Gaussian) = {worst_gauss:.3e} | "
          f"tree vs dense-Laplace (Poisson) = {worst_pois:.3e}")
    return max(worst_gauss, worst_pois)


def main():
    # 3-leaf tree with correlated A,B and independent C.
    tree = build_tree("((A:1,B:1):1,C:2):0;")
    C = bm_covariance(tree)
    print("BM covariance:\n", C)

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(6):
        mu = rng.uniform(-0.5, 1.5)
        sigma2 = rng.uniform(0.2, 1.0)
        S = rng.uniform(5.0, 30.0, size=3)
        # Simulate latent then counts so values are self-consistent.
        z = rng.multivariate_normal(mu * np.ones(3), sigma2 * C)
        Y = rng.poisson(S * np.exp(z))
        obs = PoissonObservation(Y, S)

        mean = mu * np.ones(3)
        Sigma = sigma2 * C
        lap = laplace_marginal_loglik(obs, mean, Sigma)
        bf = brute_force_logmarginal(obs, mean, Sigma)
        err = abs(lap - bf)
        worst = max(worst, err)
        print(f"trial {trial}: Y={Y.tolist()} mu={mu:.2f} s2={sigma2:.2f} | "
              f"laplace={lap:.4f} brute={bf:.4f} |diff|={err:.4f}")

    print()
    tol = 0.3
    if worst < tol:
        print(f"PASS: dense Laplace within {worst:.4f} of brute-force (tol {tol})")
    else:
        print(f"CHECK: worst |diff|={worst:.4f} exceeds {tol} (Laplace is approximate)")

    print()
    print("O(n) tree-Laplace correctness (BM / OU / OU2):")
    worst_tree = 0.0
    for nm, nwk in [("8tip", "(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):0.01;"),
                    ("balanced-64", balanced_newick(6))]:
        worst_tree = max(worst_tree, check_tree_laplace(nm, nwk))
    tree_tol = 1e-6
    print()
    if worst_tree < tree_tol:
        print(f"PASS: tree-Laplace matches its oracles (worst {worst_tree:.3e} < {tree_tol:g})")
    else:
        print(f"FAIL: tree-Laplace worst error {worst_tree:.3e} exceeds {tree_tol:g}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
