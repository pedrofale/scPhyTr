"""Ground-truth comparison of two Poisson evolutionary models for gene-gene corr.

Both models put a latent log-rate per gene on the tree and a Poisson likelihood on
raw counts; they differ only in the *structure of the latent diffusion*:

  * gene-level BM  -- a full p x p diffusion K (each gene its own latent BM,
    correlations free): ``fit_mv_em`` + ``PoissonObservation`` (O(n p^3)).
  * factor-level BM -- k independent factor BMs mapped through loadings W, so the
    gene-gene diffusion is the rank-k ``K = W W^T``:
    ``fit_poisson_factor_analysis`` (O(N k^3)).

Ground truth is generated from a *known* gene-gene covariance ``K_true`` via a
matrix-normal latent (row cov = tree C, column cov = K_true), then Poisson counts
-- a generator that privileges neither parameterization. We compare how well each
estimator recovers the true gene-gene correlation matrix ``corr(K_true)`` in two
regimes: a low-rank truth (real programs) and a full-rank truth.

Run: ``python -m docs.figures.compare_genelevel_vs_factor``.
"""
import warnings; warnings.filterwarnings("ignore")
import random
import time

import numpy as np
import ete3

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance
from scphytr.inference.laplace import PoissonObservation
from scphytr.tools.em import fit_mv_em
from scphytr.tools.estimation import cov_to_corr
from scphytr.tools.poisson_factor import fit_poisson_factor_analysis


def wrap(et):
    T = Tree(); T.phylotree = et; T.root = et.get_tree_root()
    return T


def random_tree(n, seed=0):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    t = ete3.Tree(); t.populate(n)
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


# --------------------------------------------------------------------------- #
# Ground-truth covariances and the matrix-normal Poisson generator
# --------------------------------------------------------------------------- #

def K_lowrank(p, k, rng, var=0.6):
    """Rank-k gene-gene covariance: a few overlapping programs (+ tiny ridge)."""
    W = np.zeros((p, k))
    for j in range(k):
        idx = rng.choice(p, size=p // 2, replace=False)
        W[idx, j] = rng.uniform(0.5, 1.0, p // 2) * (1 if rng.random() < 0.5 else -1)
    K = W @ W.T
    K *= var / np.mean(np.diag(K))
    return K + 1e-3 * np.eye(p)


def K_fullrank(p, rng, var=0.6, rho=0.7):
    """Full-rank gene-gene covariance: AR(1)-style decaying correlations."""
    idx = np.arange(p)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    d = rng.uniform(0.7, 1.3, p)
    K = (np.outer(d, d) ** 0.5) * R
    K *= var / np.mean(np.diag(K))
    return K + 1e-6 * np.eye(p)


def simulate_counts(tree, K_true, mu, sizes, seed=0):
    """Matrix-normal latent log-rates (row cov C, col cov K_true) -> Poisson counts."""
    rng = np.random.default_rng(seed)
    C = bm_covariance(tree)
    n, p = C.shape[0], K_true.shape[0]
    LC = np.linalg.cholesky(C + 1e-8 * np.eye(n))
    LK = np.linalg.cholesky(K_true)
    Z = mu[None, :] + LC @ rng.standard_normal((n, p)) @ LK.T
    lam = sizes[:, None] * np.exp(Z)
    return rng.poisson(lam).astype(float)


# --------------------------------------------------------------------------- #
# Recovery metrics on the off-diagonal correlations
# --------------------------------------------------------------------------- #

def offdiag_recovery(R_est, R_true):
    iu = np.triu_indices_from(R_true, k=1)
    a, b = R_est[iu], R_true[iu]
    r = np.corrcoef(a, b)[0, 1]
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return r, rmse


def run_scenario(name, K_true, n, p, k, seed, depth=3000.0):
    tree = wrap(random_tree(n, seed=seed))
    names = [l.name for l in tree.root.get_leaves()]
    rng = np.random.default_rng(7000 + seed)
    mu = rng.uniform(-5.0, -3.0, p)
    sizes = rng.gamma(4.0, depth / 4.0, n)
    Y = simulate_counts(tree, K_true, mu, sizes, seed=seed)
    R_true = cov_to_corr(K_true)

    obs = PoissonObservation(Y, offsets=sizes)

    t0 = time.time()
    fit_gl = fit_mv_em(tree, obs, model="BM", max_em=40, tol=1e-3)
    t_gl = time.time() - t0
    R_gl = cov_to_corr(fit_gl.K)

    t0 = time.time()
    fit_fa = fit_poisson_factor_analysis(Y, tree, k=k, sizes=sizes, leaf_names=names, n_iter=40)
    t_fa = time.time() - t0
    R_fa = fit_fa.evolutionary_correlation()

    # regularized gene-level: K = W W^T + diag(d) (factor + idiosyncratic heritable)
    t0 = time.time()
    fit_rg = fit_mv_em(tree, obs, model="BM", k_factor=k, max_em=80, tol=1e-4)
    t_rg = time.time() - t0
    R_rg = cov_to_corr(fit_rg.K)

    r_gl, e_gl = offdiag_recovery(R_gl, R_true)
    r_fa, e_fa = offdiag_recovery(R_fa, R_true)
    r_rg, e_rg = offdiag_recovery(R_rg, R_true)
    return {"name": name, "zeros": float(np.mean(Y == 0)),
            "r_gl": r_gl, "e_gl": e_gl, "t_gl": t_gl,
            "r_fa": r_fa, "e_fa": e_fa, "t_fa": t_fa,
            "r_rg": r_rg, "e_rg": e_rg, "t_rg": t_rg}


def main(n=120, p=16, k=3, seeds=(0, 1, 2)):
    print(f"Gene-level (full K) vs factor-level (K=WW^T) Poisson BM "
          f"-- gene-gene corr recovery (n={n}, p={p}, k={k})\n")
    truths = [("LOW-RANK truth (rank k)", lambda r: K_lowrank(p, k, r)),
              ("FULL-RANK truth (AR1) ", lambda r: K_fullrank(p, r))]
    depths = [("deep  ~3000 UMI", 3000.0), ("shallow ~120 UMI", 120.0)]
    for tag, Kfn in truths:
        for dname, depth in depths:
            rows = []
            for s in seeds:
                K_true = Kfn(np.random.default_rng(100 + s))
                rows.append(run_scenario(tag, K_true, n, p, k, seed=s, depth=depth))
            def avg(key):
                return float(np.mean([row[key] for row in rows]))
            best = max([("gene-level", avg("r_gl")), ("factor", avg("r_fa")),
                        ("factor+idio", avg("r_rg"))], key=lambda kv: kv[1])[0]
            print(f"{tag} | {dname} (zeros~{avg('zeros'):.2f})   -> best: {best}")
            print(f"   gene-level (full K)     : r={avg('r_gl'):.3f}  rmse={avg('e_gl'):.3f}  ({avg('t_gl'):.1f}s)")
            print(f"   factor (WW^T)           : r={avg('r_fa'):.3f}  rmse={avg('e_fa'):.3f}  ({avg('t_fa'):.1f}s)")
            print(f"   factor+idio (WW^T+D)    : r={avg('r_rg'):.3f}  rmse={avg('e_rg'):.3f}  ({avg('t_rg'):.1f}s)")
            print()


if __name__ == "__main__":
    main()
