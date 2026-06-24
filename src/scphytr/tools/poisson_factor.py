"""Poisson phylogenetic factor analysis: a low-rank latent model for raw counts.

This is the count analogue of :mod:`scphytr.tools.factor_analysis`. Instead of a
Gaussian readout it places a Poisson likelihood on raw UMI counts, with a small
number ``k`` of latent factors that evolve along the tree:

    z_i in R^k        factor scores at leaf i; each factor an independent
                      Brownian motion on the tree  (cov C, the deconfounding prior)
    eta_ig = mu_g + (W z_i)_g          per-gene log-rate (W is p x k loadings)
    y_ig  ~ Poisson(s_i * exp(eta_ig)) s_i a per-cell size (library) offset.

We estimate ``W`` (the gene programs) and ``mu`` by **maximum likelihood**,
integrating out the latent factors ``z`` -- the evolutionary structure lives
entirely in the factors' tree prior, so the programs are deconfounded from shared
ancestry exactly as in the Gaussian PFA, but now for counts.

Inference reuses the multivariate latent-tree Laplace machinery
(:mod:`scphytr.inference.tree_laplace_mv`) with the *per-node latent dimension set
to k* (the factors, prior precision ``A (x) I_k``). The Poisson leaf likelihood
couples the k factors through ``W`` -- a full ``k x k`` observation curvature --
which the generalized block elimination handles in O(N k^3). Fitting is Laplace-EM:

  * **E-step**  -- Laplace posterior of the factor scores (mode + covariances) via
    the tree smoother (block Felsenstein pruning / RTS).
  * **M-step**  -- update ``(W, mu)`` by maximizing the expected Poisson
    log-likelihood with the latent-uncertainty correction
    ``E[exp(eta_ig)] = exp(mu_g + W_g zbar_i + 1/2 W_g Sigma_i W_g^T)``. This is
    *concave in (mu_g, W_g) per gene*, so the M-step is fast and robust.

Naive (phylogeny-ignoring) factor analysis is the same fit on a star tree (all
leaves off the root by unit branches), where the factor prior is i.i.d. across
cells; passing such a tree gives the deconfounding-free baseline.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ..inference.tree_laplace_mv import _MVTreeModel


# --------------------------------------------------------------------------- #
# Leaf likelihood
# --------------------------------------------------------------------------- #

class _PoissonFactorLeafObs:
    """y_ig ~ Poisson(s_i exp(mu_g + (W z_i)_g)); exposes derivatives in z (n, k)."""

    def __init__(self, Y, W, mu, sizes):
        self.Y = np.asarray(Y, dtype=float)        # (n, p) counts, leaf order
        self.W = np.asarray(W, dtype=float)        # (p, k)
        self.mu = np.asarray(mu, dtype=float)      # (p,)
        self.s = np.asarray(sizes, dtype=float)    # (n,)
        self.n, self.p = self.Y.shape
        self.k = self.W.shape[1]

    def _lam(self, F):
        eta = self.mu[None, :] + F @ self.W.T      # (n, p)
        return self.s[:, None] * np.exp(eta)

    def loglik(self, F):
        eta = self.mu[None, :] + F @ self.W.T
        lam = self.s[:, None] * np.exp(eta)
        return float(np.sum(self.Y * eta - lam))   # drops constant log y!

    def grad(self, F):
        return (self.Y - self._lam(F)) @ self.W    # (n, k)

    def neg_hess_block(self, F):
        lam = self._lam(F)                          # (n, p)
        return np.einsum("gj,ig,gl->ijl", self.W, lam, self.W)   # (n, k, k), PSD

    def mode_init(self):
        return np.zeros((self.n, self.k))           # objective is concave in F


# --------------------------------------------------------------------------- #
# E-step: Laplace posterior of the factor scores on the tree
# --------------------------------------------------------------------------- #

def _factor_estep(M, obs, z_init=None, max_iter=50, tol=1e-9):
    """Damped-Newton Laplace mode + posterior covariances for the k-dim factors.

    Returns (Z, Sigma, marginal_loglik): ``Z`` (N, k) posterior mode over all
    nodes, ``Sigma`` (N, k, k) posterior covariance blocks, and the Laplace
    marginal log-likelihood. Mirrors ``tree_laplace_mv._newton_mode`` but with the
    full k x k leaf observation curvature.
    """
    leaf_idx = M.leaf_node_idx
    k = M.p

    def psi(Z):
        return -obs.loglik(Z[leaf_idx]) + 0.5 * M.prior_quad(Z)

    Z = np.zeros((M.N, k))
    if z_init is not None:
        Z[leaf_idx] = z_init
    for i in M.fixed:
        Z[i] = M.mu0[i]
    cur = psi(Z)

    for _ in range(max_iter):
        F = Z[leaf_idx]
        grad = M.prior_grad(Z)
        grad[leaf_idx] -= obs.grad(F)
        Wb = np.zeros((M.N, k, k))
        Wb[leaf_idx] = obs.neg_hess_block(F)
        step, _ = M.solve(Wb, -grad)
        t, new = 1.0, cur
        for _ in range(40):
            Z_try = Z + t * step
            new = psi(Z_try)
            if new <= cur:
                break
            t *= 0.5
        converged = abs(new - cur) < tol * (1.0 + abs(cur))
        Z, cur = Z_try, new
        if converged:
            break

    F = Z[leaf_idx]
    Wb = np.zeros((M.N, k, k))
    Wb[leaf_idx] = obs.neg_hess_block(F)
    log_det_QW = M.log_det(Wb)
    marg = obs.loglik(F) - 0.5 * M.prior_quad(Z) + 0.5 * M.log_det_Q - 0.5 * log_det_QW
    Sigma, _ = M.posterior_covariances(Wb)
    return Z, Sigma, marg


# --------------------------------------------------------------------------- #
# M-step: concave per-gene Poisson update of (W, mu) with uncertainty correction
# --------------------------------------------------------------------------- #

def _mstep_negll(params, Y, s, zbar, Sig, p, k):
    W = params[: p * k].reshape(p, k)
    mu = params[p * k:]
    eta = mu[None, :] + zbar @ W.T                  # (n, p)
    quad = 0.5 * jnp.einsum("gj,ijl,gl->ig", W, Sig, W)   # (n, p)
    lam = s[:, None] * jnp.exp(eta + quad)
    return -(jnp.sum(Y * eta - lam))


_mstep_vg = jax.jit(jax.value_and_grad(_mstep_negll), static_argnums=(5, 6))


def _mstep(Y, s, zbar, Sig, W0, mu0, maxiter=200):
    p, k = W0.shape
    Yj, sj, zj, Sj = (jnp.asarray(Y), jnp.asarray(s), jnp.asarray(zbar), jnp.asarray(Sig))
    x0 = np.concatenate([np.asarray(W0).ravel(), np.asarray(mu0)])

    def fun(x):
        v, g = _mstep_vg(jnp.asarray(x), Yj, sj, zj, Sj, p, k)
        return float(v), np.asarray(g, dtype=float)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8})
    return res.x[: p * k].reshape(p, k), res.x[p * k:]


# --------------------------------------------------------------------------- #
# Fitted model
# --------------------------------------------------------------------------- #

@dataclass
class FittedPoissonFactorModel:
    """A fitted Poisson phylogenetic factor model."""
    W: np.ndarray                 # (p, k) loadings (gene programs)
    mu: np.ndarray                # (p,) per-gene log-rate offset
    scores: np.ndarray            # (n, k) posterior-mean factor scores at leaves
    loglik: float                 # Laplace marginal log-likelihood
    leaf_names: list = None
    sizes: np.ndarray = field(default=None, repr=False)
    history: list = field(default_factory=list, repr=False)

    @property
    def p(self):
        return self.W.shape[0]

    @property
    def k(self):
        return self.W.shape[1]

    @property
    def n(self):
        return self.scores.shape[0]

    def n_params(self):
        p, k = self.p, self.k
        return p * k - k * (k - 1) // 2 + p      # loadings (less rotation) + mu

    def aic(self):
        return 2.0 * self.n_params() - 2.0 * self.loglik

    def bic(self):
        return self.n_params() * np.log(self.n) - 2.0 * self.loglik

    def loadings(self):
        import pandas as pd
        return pd.DataFrame(self.W, columns=[f"factor{j}" for j in range(self.k)])

    def subspace(self):
        """Orthonormal basis (p x k) of the loading column space."""
        Q, _ = np.linalg.qr(self.W)
        return Q[:, : self.k]

    def evolutionary_covariance(self):
        """Gene-gene evolutionary (diffusion) covariance ``K = W W^T`` (p x p).

        Because the k factors evolve as independent unit-rate Brownian motions,
        the gene log-rates ``eta = W z`` are a multivariate BM with diffusion
        matrix ``W W^T`` -- the rank-k, deconfounded gene-gene covariance on the
        log-rate scale. This is the low-rank counterpart of the full ``K``
        estimated by :func:`scphytr.tools.fit_mv_latent`.
        """
        return self.W @ self.W.T

    def evolutionary_correlation(self):
        """Gene-gene evolutionary correlation matrix, ``corr(W W^T)`` (p x p).

        Rotation-invariant (unlike ``W`` itself), so this is the right object to
        report for gene-gene structure. Diagonal-zero genes (no loading) are
        returned with zero correlation.
        """
        K = self.evolutionary_covariance()
        d = np.sqrt(np.clip(np.diag(K), 1e-300, None))
        R = K / np.outer(d, d)
        R[np.diag(K) <= 0] = 0.0
        R[:, np.diag(K) <= 0] = 0.0
        np.fill_diagonal(R, 1.0)
        return R


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def _init_W_mu(Y, sizes, k, rng):
    """Initialise from a quick factorization of log-normalized counts."""
    ln = np.log1p(Y / sizes[:, None] * np.median(sizes))
    mu0 = np.log(np.maximum(Y / sizes[:, None], 0).mean(0) + 1e-3)
    R = ln - ln.mean(0)
    try:
        _, s, Vt = np.linalg.svd(R, full_matrices=False)
        W0 = Vt[:k].T * (s[:k] / max(np.sqrt(len(R)), 1.0))
    except np.linalg.LinAlgError:
        W0 = 0.1 * rng.standard_normal((Y.shape[1], k))
    if W0.shape[1] < k:
        W0 = np.hstack([W0, 0.1 * rng.standard_normal((Y.shape[1], k - W0.shape[1]))])
    return W0, mu0


def fit_poisson_factor_analysis(counts, tree, k, sizes=None, leaf_names=None,
                                n_iter=50, tol=1e-5, seed=0, verbose=False):
    """Maximum-likelihood Poisson phylogenetic factor analysis.

    Parameters
    ----------
    counts : (n, p) array of raw counts. Rows are cells/leaves, columns genes.
        If ``leaf_names`` is given, rows are reordered to the tree's leaf order.
    tree : ``scphytr.utils.tree.Tree`` (or compatible) with the cell phylogeny.
        A star tree gives the phylogeny-naive baseline.
    k : number of latent factors.
    sizes : (n,) per-cell size factors (default: library size / its mean).
    n_iter, tol : Laplace-EM iteration budget and relative-loglik tolerance.

    Returns
    -------
    FittedPoissonFactorModel
    """
    counts = np.asarray(counts, dtype=float)
    order_names = tree.phylotree.get_leaf_names()
    if leaf_names is not None:
        pos = {nm: i for i, nm in enumerate(list(leaf_names))}
        idx = [pos[nm] for nm in order_names]
        Y = counts[idx]
        sizes = None if sizes is None else np.asarray(sizes, dtype=float)[idx]
    else:
        Y = counts
    n, p = Y.shape

    if sizes is None:
        lib = Y.sum(1)
        sizes = lib / np.mean(lib)
    sizes = np.asarray(sizes, dtype=float)

    # ensure a free (positive-length) root so the factor prior is proper
    if float(tree.root.dist) <= 0:
        tree.root.dist = 1.0

    M = _MVTreeModel(tree, alpha=None, theta=np.zeros(k), K=np.eye(k),
                     root_value=np.zeros(k))

    rng = np.random.default_rng(seed)
    W, mu = _init_W_mu(Y, sizes, k, rng)

    z_init = None
    prev = -np.inf
    history = []
    for it in range(n_iter):
        obs = _PoissonFactorLeafObs(Y, W, mu, sizes)
        Z, Sigma, marg = _factor_estep(M, obs, z_init=z_init)
        z_init = Z[M.leaf_node_idx]
        zbar = Z[M.leaf_node_idx]
        Sig = Sigma[M.leaf_node_idx]
        W, mu = _mstep(Y, sizes, zbar, Sig, W, mu)
        history.append(marg)
        if verbose:
            print(f"  EM iter {it:2d}: marginal loglik {marg:.2f}")
        if np.isfinite(prev) and abs(marg - prev) < tol * (1.0 + abs(prev)):
            break
        prev = marg

    # final E-step scores with the updated (W, mu)
    obs = _PoissonFactorLeafObs(Y, W, mu, sizes)
    Z, _, marg = _factor_estep(M, obs, z_init=z_init)
    scores = Z[M.leaf_node_idx]
    return FittedPoissonFactorModel(W=W, mu=mu, scores=scores, loglik=marg,
                                    leaf_names=order_names, sizes=sizes,
                                    history=history)


# --------------------------------------------------------------------------- #
# Simulation (for validation)
# --------------------------------------------------------------------------- #

def simulate_poisson_pfa(tree, W, mu, mean_size=2000.0, dynamics=None, seed=0):
    """Simulate Poisson factor data on a tree.

    ``dynamics`` is an optional length-k list of ``("BM", rate)`` / ``("OU",
    alpha, rate)`` per factor (default: unit-rate BM for every factor). Returns
    ``(Y, X, sizes, leaf_names)`` with counts ``Y`` (n, p), true factor scores
    ``X`` (n, k), and per-cell sizes.
    """
    from ..utils.covariance import bm_covariance, ou_covariance

    W = np.asarray(W, dtype=float)
    mu = np.asarray(mu, dtype=float)
    p, k = W.shape
    leaf_names = [leaf.name for leaf in tree.root.get_leaves()]
    n = len(leaf_names)
    rng = np.random.default_rng(seed)
    C = bm_covariance(tree)

    X = np.empty((n, k))
    for j in range(k):
        dyn = ("BM", 1.0) if dynamics is None else dynamics[j]
        if dyn[0] == "BM":
            Cj = dyn[1] * C
        else:
            _, alpha, rate = dyn
            Cj = rate * ou_covariance(tree, alpha)
        L = np.linalg.cholesky(Cj + 1e-10 * np.eye(n))
        X[:, j] = L @ rng.standard_normal(n)

    sizes = rng.gamma(shape=4.0, scale=mean_size / 4.0, size=n) / mean_size
    eta = mu[None, :] + X @ W.T
    lam = (sizes * mean_size)[:, None] * np.exp(eta)
    Y = rng.poisson(lam).astype(float)
    return Y, X, sizes, leaf_names
