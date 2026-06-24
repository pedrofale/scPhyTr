"""Phylogenetic factor analysis (PFA) and its phylogeny-naive counterpart.

A small number ``k`` of latent factors evolve along the tree; each cell's genes
are a linear readout of the factors at its leaf,

    y_i = W x_i + mu + eps_i,   eps_i ~ N(0, Psi)  (Psi diagonal),

with the factor trajectory ``x`` a Gaussian process on the tree. Stacking the
leaves, the marginal law of the data is Gaussian with covariance

    Cov(vec Y) = sum_j  C_j (x) w_j w_j^T   +   I_n (x) Psi,

where ``C_j`` is the n x n phylogenetic covariance of factor ``j`` and ``w_j`` is
its loading column. The *only* thing that distinguishes phylogenetic from naive
factor analysis is whether the row (cell) covariance is the tree covariance
``C`` or the identity ``I_n`` -- so the two are the same estimator with one
switch, which makes their comparison airtight.

For the Brownian-motion model all factors share ``C`` (``C_j = sigma_j^2 C``);
folding ``sigma_j`` into the loading-column norm leaves a single shared ``C`` and
a free ``W``. The marginal log-likelihood then diagonalizes in the eigenbasis of
``C``: with ``C = U diag(lambda) U^T`` and rotated data ``Ytil = U^T Y``, the
rotated rows are independent Gaussians with per-row covariance
``lambda_i W W^T + Psi``, each handled in O(p k^2 + k^3) by the Woodbury
identity. Naive factor analysis is the special case ``lambda_i == 1``.

The per-factor dynamics (which factors are BM, which OU, and the BM rates) are
recovered by running the univariate BM/OU model selection of
``scphytr.tools.model_selection`` on the inferred factor trajectories -- the
factors are themselves traits on the tree.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_pfa(tree, W, dynamics, noise_sd=0.3, mu=0.0, seed=0):
    """Simulate phylogenetic-factor-analysis data.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    W : (p, k) loading matrix.
    dynamics : list of length ``k`` describing each factor's evolution, each an
        ``("BM", rate)`` or ``("OU", alpha, rate)`` tuple. Each factor is an
        independent Gaussian process on the tree (mean 0); for OU the optimum is
        0 and the stationary-root convention is used.
    noise_sd : scalar or (p,) idiosyncratic Gaussian noise standard deviation.
    mu : scalar or (p,) gene means.

    Returns
    -------
    (Y, X_leaf, leaf_names) with Y (n, p), X_leaf (n, k) the true factor values
    at the leaves, in ``tree.root.get_leaves()`` order.
    """
    from ..utils.covariance import bm_covariance, ou_covariance

    W = np.asarray(W, dtype=float)
    p, k = W.shape
    leaf_names = [leaf.name for leaf in tree.root.get_leaves()]
    n = len(leaf_names)
    rng = np.random.default_rng(seed)
    C_bm = bm_covariance(tree)

    X = np.empty((n, k))
    for j, dyn in enumerate(dynamics):
        if dyn[0] == "BM":
            Cj = dyn[1] * C_bm
        elif dyn[0] == "OU":
            _, alpha, rate = dyn
            Cj = rate * ou_covariance(tree, alpha)
        else:
            raise ValueError(f"Unknown factor dynamics '{dyn[0]}'.")
        L = np.linalg.cholesky(Cj + 1e-10 * np.eye(n))
        X[:, j] = L @ rng.standard_normal(n)

    noise = np.asarray(noise_sd, dtype=float) * rng.standard_normal((n, p))
    Y = X @ W.T + np.asarray(mu, dtype=float) + noise
    return Y, X, leaf_names


# ---------------------------------------------------------------------------
# Marginal log-likelihood (rotated, per-row Woodbury). Shared by phylo & naive.
# ---------------------------------------------------------------------------

def _unpack(params, p, k):
    W = params[: p * k].reshape(p, k)
    log_psi = params[p * k : p * k + p]
    mu = params[p * k + p :]
    return W, log_psi, mu


def _neg_loglik(params, Ytil, lam, mbar, p, k):
    """Negative Gaussian log-likelihood in the rotated (cell-eigen) basis.

    Ytil : (n, p)  rotated data  U^T Y
    lam  : (n,)    row variances (eigenvalues of the row covariance; ones=naive)
    mbar : (n,)    U^T 1_n  (the rotated all-ones vector; the per-row mean scale)
    """
    W, log_psi, mu = _unpack(params, p, k)
    psi = jnp.exp(log_psi)                       # (p,)
    R = Ytil - mbar[:, None] * mu[None, :]       # (n, p) residuals
    Wt_psi = W.T / psi[None, :]                  # (k, p)  = W^T Psi^{-1}
    A = Wt_psi @ W                               # (k, k)  = W^T Psi^{-1} W
    Ik = jnp.eye(k)
    log_det_psi = jnp.sum(log_psi)
    rinv = R / psi[None, :]                       # (n, p)  Psi^{-1} r
    quad_diag = jnp.sum(R * rinv, axis=1)         # (n,)    r^T Psi^{-1} r
    b = R @ Wt_psi.T                              # (n, k)  W^T Psi^{-1} r

    def row(lam_i, qd_i, b_i):
        Ki = Ik + lam_i * A                       # (k, k)
        Lk = jnp.linalg.cholesky(Ki)
        sol = jax.scipy.linalg.cho_solve((Lk, True), b_i)
        quad = qd_i - lam_i * (b_i @ sol)
        log_det = log_det_psi + 2.0 * jnp.sum(jnp.log(jnp.diag(Lk)))
        return quad + log_det

    terms = jax.vmap(row)(lam, quad_diag, b)      # (n,)
    n = Ytil.shape[0]
    return 0.5 * (jnp.sum(terms) + n * p * jnp.log(2.0 * jnp.pi))


_value_and_grad = jax.jit(jax.value_and_grad(_neg_loglik), static_argnums=(4, 5))


# ---------------------------------------------------------------------------
# Fitted model
# ---------------------------------------------------------------------------

@dataclass
class FittedFactorModel:
    """A fitted (phylogenetic or naive) factor-analysis model."""
    W: np.ndarray                 # (p, k) loadings (BM rates folded into norms)
    psi: np.ndarray               # (p,) idiosyncratic variances
    mu: np.ndarray                # (p,) gene means
    loglik: float
    phylogenetic: bool
    U: np.ndarray = field(repr=False)        # (n, n) eigenvectors of row cov
    lam: np.ndarray = field(repr=False)      # (n,) eigenvalues of row cov
    mbar: np.ndarray = field(repr=False)     # (n,) U^T 1
    Ytil: np.ndarray = field(repr=False)     # (n, p) rotated data
    leaf_names: list = None

    @property
    def n(self):
        return self.Ytil.shape[0]

    @property
    def p(self):
        return self.W.shape[0]

    @property
    def k(self):
        return self.W.shape[1]

    def n_params(self):
        # loadings minus rotational gauge freedom, plus Psi and mu.
        p, k = self.p, self.k
        return p * k - k * (k - 1) // 2 + p + p

    def aic(self):
        return 2.0 * self.n_params() - 2.0 * self.loglik

    def bic(self):
        return self.n_params() * np.log(self.n) - 2.0 * self.loglik

    def loadings(self):
        return pd.DataFrame(self.W, columns=[f"factor{j}" for j in range(self.k)])

    def subspace(self):
        """Orthonormal basis (p x k) for the column space of the loadings."""
        Q, _ = np.linalg.qr(self.W)
        return Q[:, : self.k]

    def factor_scores(self):
        """Posterior mean factor values at each leaf, (n, k), in leaf order.

        Exact for this linear-Gaussian model: in the cell-eigenbasis the rotated
        factor scores are the standard factor-analysis posterior means with the
        per-row prior variance ``lambda_i``; rotating back by ``U`` returns the
        leaf-ordered trajectories.
        """
        W, psi, lam = self.W, self.psi, self.lam
        R = self.Ytil - self.mbar[:, None] * self.mu[None, :]   # (n, p)
        Wt_psi = W.T / psi[None, :]                              # (k, p)
        A = Wt_psi @ W                                           # (k, k)
        b = R @ Wt_psi.T                                         # (n, k) = W^T Psi^{-1} r
        Xtil = np.empty((self.n, self.k))
        Ik = np.eye(self.k)
        for i in range(self.n):
            Ki = Ik + lam[i] * A
            Xtil[i] = lam[i] * np.linalg.solve(Ki, b[i])
        return self.U @ Xtil                                     # (n, k), leaf order


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _init_params(Ytil, mbar, p, k, rng):
    """Initialise from a quick PCA of the (rotated) data."""
    mu0 = (Ytil * mbar[:, None]).sum(0) / (mbar ** 2).sum()
    R = Ytil - mbar[:, None] * mu0[None, :]
    # SVD-based loading init; scale columns by singular values.
    try:
        _, s, Vt = np.linalg.svd(R, full_matrices=False)
        W0 = (Vt[:k].T * (s[:k] / max(np.sqrt(len(R)), 1.0)))
    except np.linalg.LinAlgError:
        W0 = 0.1 * rng.standard_normal((p, k))
    if W0.shape[1] < k:
        W0 = np.hstack([W0, 0.1 * rng.standard_normal((p, k - W0.shape[1]))])
    resid = R - R @ np.linalg.pinv(W0).T @ W0.T if k > 0 else R
    psi0 = np.maximum(resid.var(axis=0), 1e-2)
    return np.concatenate([W0.ravel(), np.log(psi0), mu0])


def fit_factor_analysis(Y, row_cov=None, k=2, restarts=2, seed=0,
                        leaf_names=None, maxiter=2000):
    """Fit a ``k``-factor model to ``Y`` (n x p).

    Parameters
    ----------
    Y : (n, p) array. Rows are cells/leaves, columns are genes.
    row_cov : (n, n) array or None. The cell (row) covariance. ``None`` gives
        the phylogeny-*naive* model (rows i.i.d.); passing the tree covariance
        ``C`` gives phylogenetic factor analysis. The two differ *only* here.
    k : number of latent factors.

    Returns
    -------
    FittedFactorModel
    """
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape
    rng = np.random.default_rng(seed)

    if row_cov is None:
        U = np.eye(n)
        lam = np.ones(n)
    else:
        row_cov = np.asarray(row_cov, dtype=float)
        w, U = np.linalg.eigh(0.5 * (row_cov + row_cov.T))
        lam = np.clip(w, 1e-10, None)
    Ytil = U.T @ Y
    mbar = U.T @ np.ones(n)

    best = None
    for r in range(restarts + 1):
        x0 = _init_params(Ytil, mbar, p, k, rng)
        if r > 0:
            x0 = x0 + 0.1 * rng.standard_normal(x0.shape)

        def fun(x):
            v, g = _value_and_grad(jnp.asarray(x), jnp.asarray(Ytil),
                                   jnp.asarray(lam), jnp.asarray(mbar), p, k)
            return float(v), np.asarray(g, dtype=float)

        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res

    W, log_psi, mu = _unpack(best.x, p, k)
    return FittedFactorModel(
        W=np.asarray(W), psi=np.exp(np.asarray(log_psi)), mu=np.asarray(mu),
        loglik=float(-best.fun), phylogenetic=row_cov is not None,
        U=U, lam=lam, mbar=mbar, Ytil=Ytil, leaf_names=leaf_names)


def fit_phylo_factor_analysis(tree, Y, k=2, model="BM", alpha=None, **kwargs):
    """Convenience wrapper: build the tree row covariance and fit PFA.

    ``model="BM"`` uses the Brownian phylogenetic covariance; ``model="OU"``
    uses the OU covariance at ``alpha`` (a shared mean-reversion for the row
    structure). ``Y`` must be in ``tree.root.get_leaves()`` order.
    """
    from ..utils.covariance import bm_covariance, ou_covariance

    if model == "BM":
        C = bm_covariance(tree)
    elif model == "OU":
        if alpha is None:
            raise ValueError("model='OU' requires `alpha`.")
        C = ou_covariance(tree, alpha)
    else:
        raise ValueError(f"Unknown model '{model}'.")
    names = [leaf.name for leaf in tree.root.get_leaves()]
    return fit_factor_analysis(Y, row_cov=C, k=k, leaf_names=names, **kwargs)


# ---------------------------------------------------------------------------
# Subspace comparison (loadings are identified only up to a k x k rotation)
# ---------------------------------------------------------------------------

def principal_angles(A, B):
    """Principal angles (radians) between the column spaces of ``A`` and ``B``."""
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=float))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=float))
    kk = min(Qa.shape[1], Qb.shape[1])
    s = np.linalg.svd(Qa[:, :kk].T @ Qb[:, :kk], compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_error(A, B):
    """Grassmann distance ``sqrt(sum sin^2 theta)`` between two column spaces.

    Zero iff the loading subspaces coincide; rotation-invariant, so it is the
    right way to compare factor-analysis loadings (which are only identified up
    to a rotation of the factors).
    """
    theta = principal_angles(A, B)
    return float(np.sqrt(np.sum(np.sin(theta) ** 2)))


def procrustes_align(W_est, W_true):
    """Rotate/reflect ``W_est`` to best match ``W_true`` (orthogonal Procrustes).

    Returns the aligned loadings; useful for plotting columns side by side.
    """
    W_est = np.asarray(W_est, dtype=float)
    W_true = np.asarray(W_true, dtype=float)
    M = W_est.T @ W_true
    Uo, _, Vt = np.linalg.svd(M)
    Rrot = Uo @ Vt
    return W_est @ Rrot


# ---------------------------------------------------------------------------
# Per-factor dynamics: which factors are BM, which OU, and the BM rates.
# ---------------------------------------------------------------------------

def _ou_cov_jax(alpha, T, S):
    """OU phylogenetic covariance (unit diffusion) in JAX; -> BM (=S) as alpha->0."""
    expo = jnp.exp(-alpha * (T[:, None] + T[None, :] - 2.0 * S))
    # (1 - e^{-2 a S}) / (2 a), stable as a -> 0 via expm1.
    fac = -jnp.expm1(-2.0 * alpha * S) / (2.0 * alpha)
    return expo * fac


def _pfa_dyn_negloglik(x, Y, T, S, config, p, k, alpha_max):
    """Negative marginal log-likelihood for PFA with per-factor dynamics.

    Builds the dense (np x np) covariance ``sum_j C_j (x) w_j w_j^T + I (x) Psi``
    where ``C_j`` is the BM (=S) or OU covariance of factor ``j`` per ``config``.
    Used only on moderate trees; identifies *individual* factors because
    heterogeneous dynamics break factor-analysis' rotational symmetry.
    """
    n = Y.shape[0]
    W = x[: p * k].reshape(p, k)
    log_psi = x[p * k : p * k + p]
    mu = x[p * k + p : p * k + 2 * p]
    log_alpha = x[p * k + 2 * p :]
    psi = jnp.exp(log_psi)

    Sigma = jnp.kron(jnp.eye(n), jnp.diag(psi))
    oa = 0
    for j in range(k):
        wj = W[:, j]
        outer = jnp.outer(wj, wj)
        if config[j] == "OU":
            alpha = jnp.clip(jnp.exp(log_alpha[oa]), 1e-4, alpha_max)
            oa += 1
            Cj = _ou_cov_jax(alpha, T, S)
        else:
            Cj = S
        # Normalize to unit mean tip-variance: the factor's marginal variance is
        # carried by ||w_j||^2 and alpha shapes only the temporal autocorrelation.
        # This removes the OU (rate, alpha) non-identifiability that otherwise lets
        # the optimizer trade variance against mean-reversion and degenerate.
        Cj = Cj / jnp.mean(jnp.diag(Cj))
        Sigma = Sigma + jnp.kron(Cj, outer)

    r = (Y - mu[None, :]).reshape(-1)
    L = jnp.linalg.cholesky(Sigma + 1e-8 * jnp.eye(n * p))
    sol = jax.scipy.linalg.cho_solve((L, True), r)
    quad = r @ sol
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return 0.5 * (quad + log_det + n * p * jnp.log(2.0 * jnp.pi))


@dataclass
class FittedDynamicFactorModel:
    """PFA with per-factor evolutionary dynamics (individually identified)."""
    W: np.ndarray
    psi: np.ndarray
    mu: np.ndarray
    config: tuple              # per-factor dynamics ("BM"/"OU")
    alphas: np.ndarray        # per-factor OU alpha (nan for BM factors)
    loglik: float
    n: int
    leaf_names: list = None

    @property
    def p(self):
        return self.W.shape[0]

    @property
    def k(self):
        return self.W.shape[1]

    def n_params(self):
        p, k = self.p, self.k
        # No rotational gauge to subtract: heterogeneous dynamics fix the factors.
        n_alpha = sum(1 for c in self.config if c == "OU")
        return p * k + p + p + n_alpha

    def aic(self):
        return 2.0 * self.n_params() - 2.0 * self.loglik

    def bic(self):
        return self.n_params() * np.log(self.n) - 2.0 * self.loglik

    def rates(self):
        """Marginal tip-variance carried by each factor (squared loading norm).

        Because the per-factor tree covariance is normalized to unit mean
        tip-variance, ``||w_j||^2`` is the variance the factor contributes at the
        tips -- comparable across BM and OU factors.
        """
        return np.sum(self.W ** 2, axis=0)

    def summary(self):
        rate = self.rates()
        return pd.DataFrame({
            "dynamics": list(self.config),
            "alpha": self.alphas,
            "rate": rate,
        }, index=[f"factor{j}" for j in range(self.k)])


def _fit_pfa_config(tree, Y, config, T, S, restarts=2, seed=0, maxiter=1500):
    """Fit PFA with a fixed per-factor dynamics ``config`` (dense, moderate n)."""
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape
    k = len(config)
    n_alpha = sum(1 for c in config if c == "OU")
    alpha_max = 30.0 / max(float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist), 1e-12)
    rng = np.random.default_rng(seed)
    Tj, Sj = jnp.asarray(T), jnp.asarray(S)
    Yj = jnp.asarray(Y)

    vg = jax.jit(jax.value_and_grad(_pfa_dyn_negloglik),
                 static_argnums=(4, 5, 6, 7))

    # PCA-ish init.
    mu0 = Y.mean(0)
    R = Y - mu0
    _, s, Vt = np.linalg.svd(R, full_matrices=False)
    W0 = (Vt[:k].T * (s[:k] / np.sqrt(n)))
    psi0 = np.maximum(R.var(0), 1e-2)
    base = np.concatenate([W0.ravel(), np.log(psi0), mu0,
                           np.log(np.full(n_alpha, 1.0))]) if n_alpha else \
        np.concatenate([W0.ravel(), np.log(psi0), mu0])

    best = None
    for r in range(restarts + 1):
        x0 = base + (0.0 if r == 0 else 0.1 * rng.standard_normal(base.shape))

        def fun(x):
            v, g = vg(jnp.asarray(x), Yj, Tj, Sj, config, p, k, alpha_max)
            return float(v), np.asarray(g, dtype=float)

        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res

    x = best.x
    W = x[: p * k].reshape(p, k)
    psi = np.exp(x[p * k : p * k + p])
    mu = x[p * k + p : p * k + 2 * p]
    log_alpha = x[p * k + 2 * p :]
    alphas = np.full(k, np.nan)
    oa = 0
    for j in range(k):
        if config[j] == "OU":
            alphas[j] = float(np.clip(np.exp(log_alpha[oa]), 1e-4, alpha_max))
            oa += 1
    names = [leaf.name for leaf in tree.root.get_leaves()]
    return FittedDynamicFactorModel(W=np.asarray(W), psi=psi, mu=np.asarray(mu),
                                    config=tuple(config), alphas=alphas,
                                    loglik=float(-best.fun), n=n, leaf_names=names)


def detect_factor_dynamics(tree, Y, k, criterion="aic", restarts=2, seed=0):
    """Jointly fit PFA and select each factor's dynamics (BM vs OU).

    Enumerates the ``2^k`` per-factor BM/OU configurations, fits each by
    maximizing the exact marginal likelihood, and picks the configuration with
    the best information criterion. Because heterogeneous dynamics break the
    rotational symmetry of factor analysis, the returned factors are
    individually identified; ``.summary()`` reports each factor's dynamics,
    OU ``alpha``, and tip-variance. Dense O((np)^3) -- for moderate trees.

    .. warning::
        **Prototype.** This joint estimator is reliable only when factors are
        well separated (clearly different variance or strong dynamics). It has
        two known failure modes documented in ``docs/03_factor_analysis.md``:
        (i) low BM-vs-OU power for n <= ~64 (a BM trajectory can read as weak
        OU), and (ii) a degenerate likelihood mode where one loading collapses
        (rank reduction) at a lower NLL than the truth. Validate against a
        simulation before trusting the labels; see the open work in the docs.
    """
    from ..utils.covariance import phylo_times
    import itertools

    _, T, S = phylo_times(tree)
    score = {"aic": lambda m: m.aic(), "aicc": lambda m: m.aic(), "bic": lambda m: m.bic()}[criterion]

    fits = []
    for config in itertools.product(("BM", "OU"), repeat=k):
        fits.append(_fit_pfa_config(tree, Y, config, T, S, restarts=restarts, seed=seed))
    best = min(fits, key=score)
    return best, fits


def classify_factor_dynamics(tree, fitted, criterion="aic", leaf_names=None):
    """Classify each inferred factor as BM or OU and report its rate.

    Runs the univariate BM-vs-OU model selection of ``model_selection`` on each
    factor's posterior trajectory (the factor values at the leaves). Returns a
    tidy table with the selected dynamics, the criterion gap, the BM rate
    (loading-folded ``sigma^2``) and the OU ``alpha``/``theta`` where relevant.
    """
    from .model_selection import fit_bm, fit_ou, select_model

    scores = fitted.factor_scores()                      # (n, k), leaf order
    if leaf_names is None:
        leaf_names = fitted.leaf_names or [leaf.name for leaf in tree.root.get_leaves()]

    rows = []
    for j in range(fitted.k):
        values = {name: float(scores[i, j]) for i, name in enumerate(leaf_names)}
        bm = fit_bm(tree, values)
        ou = fit_ou(tree, values)
        best, sc = select_model([bm, ou], criterion=criterion)
        rows.append({
            "factor": f"factor{j}",
            "selected": best.name,
            "adaptive": best.name == "OU",
            f"d_{criterion}": abs(sc["BM"] - sc["OU"]),
            "rate_BM": bm.params["sigma2"],
            "alpha_OU": ou.params["alpha"],
            "theta_OU": ou.params["theta"],
            "loglik_BM": bm.loglik,
            "loglik_OU": ou.loglik,
        })
    return pd.DataFrame(rows).set_index("factor")
