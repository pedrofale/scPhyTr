"""Compare E-step inference engines (Laplace / importance sampling / NUTS).

The EM E-step computes the posterior over the latent tree and the expected
sufficient statistics the M-step consumes. These can be produced by different
inference engines; the M-step is identical regardless. Here we check, on the
Poisson/BM running example, that the engines agree at a fixed eta, that the
tree simulation smoother matches the analytic posterior covariance, and that
end-to-end EM recovers the planted correlation with every engine.
"""

import time
import numpy as np

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance
from scphytr.inference.laplace import PoissonObservation
from scphytr.inference.tree_laplace_mv import _MVTreeModel, mv_tree_laplace_marginal
from scphytr.inference.estep import (
    LaplaceEStep, ImportanceSamplingEStep, MCMCEStep,
)
from scphytr.tools.em import fit_mv_em


def balanced_newick(depth, root_branch=0.5):
    c = [0]

    def rec(d, root):
        if d == 0:
            c[0] += 1
            return f"L{c[0]}:1.0"
        s = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return s if root else s + ":1.0"

    return rec(depth, True) + f":{root_branch};"


def corr_cov(rho, sds):
    R = np.array([[1.0, rho], [rho, 1.0]])
    D = np.diag(sds)
    return D @ R @ D


def simulate(seed=0, depth=5):
    rng = np.random.default_rng(seed)
    tree = Tree(balanced_newick(depth, root_branch=0.5))
    n = len(tree.root.get_leaves())
    K_true = corr_cov(0.8, [0.9, 1.1])
    mu = np.array([2.0, 1.6])
    C = bm_covariance(tree)
    Z = rng.multivariate_normal(np.tile(mu, n), np.kron(C, K_true)).reshape(n, 2)
    S = rng.uniform(40, 80, size=n)
    Y = rng.poisson(S[:, None] * np.exp(Z))
    return tree, PoissonObservation(Y, S), K_true, mu, n


def edge_suffstat(es):
    """E[Δz Δzᵀ] summed over edges -- the quantity the M-step actually uses."""
    M, Z, Sig, cross = es["M"], es["Z"], es["Sigma"], es["cross"]
    tot = np.zeros((M.p, M.p))
    for i in range(M.N):
        pp = M.solve_parent[i]
        if pp < 0:
            continue
        phi = M.phi[i]
        m, mpa = Z[i], Z[pp]
        Ezz = Sig[i] + np.outer(m, m)
        Ezpa = cross[i] + np.outer(m, mpa)
        Epapa = Sig[pp] + np.outer(mpa, mpa)
        d = m - phi * mpa - M.c[i]
        quad = Ezz - phi * (Ezpa + Ezpa.T) + phi ** 2 * Epapa
        tot += quad - np.outer(d, M.c[i]) - np.outer(M.c[i], d) + np.outer(M.c[i], M.c[i])
    return tot


def main():
    tree, obs, K_true, mu, n = simulate(depth=5)     # 32 leaves
    p = 2
    print(f"tree: n={n} leaves, p={p} genes; latent BM (true corr +0.80), Poisson counts\n")

    # Evaluate every engine's E-step at the SAME (true) parameters.
    alpha, theta, K = 0.0, mu, K_true

    print("== sampler vs analytic smoother (covariance of the Laplace Gaussian) ==")
    M = _MVTreeModel(tree, alpha, theta, K)
    from scphytr.inference.tree_laplace_mv import _newton_mode
    mode = _newton_mode(M, obs)
    Wdiag = np.zeros((M.N, p)); Wdiag[M.leaf_node_idx] = obs.neg_hess_diag(mode[M.leaf_node_idx])
    Sig_an, _ = M.posterior_covariances(Wdiag)
    draws = M.sample_gaussian(Wdiag, mode, 20000, np.random.default_rng(0))
    Sig_emp = np.einsum("snp,snq->npq", draws - draws.mean(0), draws - draws.mean(0)) / draws.shape[0]
    err = np.max(np.abs(Sig_an - Sig_emp))
    print(f"  max|Sigma_analytic - Sigma_sampled| over all nodes = {err:.4f}\n")

    engines = {
        "laplace": LaplaceEStep(),
        "is":      ImportanceSamplingEStep(n_samples=8000, seed=0),
        "mcmc":    MCMCEStep(n_samples=4000, n_warmup=1000, seed=0),
    }
    laplace_logZ = mv_tree_laplace_marginal(tree, obs, alpha, theta, K)
    print(f"Laplace marginal logZ = {laplace_logZ:.3f}\n")

    print("== E-step agreement at fixed eta ==")
    ref = engines["laplace"].run(tree, obs, alpha, theta, K)
    ref_mode, ref_ss = ref["Z"], edge_suffstat(ref)
    for name, eng in engines.items():
        t0 = time.perf_counter()
        es = eng.run(tree, obs, alpha, theta, K)
        dt = time.perf_counter() - t0
        dmean = np.max(np.abs(es["Z"] - ref_mode))
        dss = np.max(np.abs(edge_suffstat(es) - ref_ss))
        extra = ""
        if "logZ" in es:
            extra += f" logZ={es['logZ']:.3f} (Δ={es['logZ']-laplace_logZ:+.3f})"
        if "ess" in es:
            extra += f" ess={es['ess']:.0f}"
        if "accept" in es:
            extra += f" accept={es['accept']:.2f}"
        print(f"  {name:8s}: max|Δmean|={dmean:.4f}  max|Δ E[ΔzΔzᵀ]|={dss:.4f}  ({dt:.1f}s){extra}")

    print("\n== end-to-end EM recovery (each engine drives the same M-step) ==")
    for name in ("laplace", "is", "mcmc"):
        eng = {"laplace": "laplace",
               "is": ImportanceSamplingEStep(n_samples=4000, seed=0),
               "mcmc": MCMCEStep(n_samples=1500, n_warmup=600, seed=0)}[name]
        t0 = time.perf_counter()
        fit = fit_mv_em(tree, obs, model="BM", trait_names=["g0", "g1"],
                        max_em=15, estep=eng)
        dt = time.perf_counter() - t0
        rho = fit.correlation().iloc[0, 1]
        print(f"  {name:8s}: corr={rho:+.3f} (true +0.80)  logL={fit.loglik:.2f}  "
              f"{fit.extra['em_iters']} iters  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
