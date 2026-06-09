"""Multivariate BM/OU parameter estimation: evolutionary rates, correlations, optima.

The evolutionary parameters live in a *latent* multivariate process on the tree:
each node carries a p-vector that evolves by BM/OU with a full diffusion matrix
``K`` (p x p). ``K``'s diagonal is the per-gene evolutionary rate and its
off-diagonals are the evolutionary covariances, so ``cov_to_corr(K)`` gives the
evolutionary correlations between genes. There are two estimation paths:

1. Directly-observed (conjugate) traits. The latent equals the observed trait, the
   covariance is separable ``K ⊗ C`` (``C`` the phylogenetic covariance), and
   Felsenstein's contrasts whiten ``C`` so the contrasts are i.i.d. ``N(0, K)``.
   The MLE of ``K`` is then just the contrast sample covariance, in O(n p^2) via
   the pruning recursion (``fit_bm_mv`` / ``fit_ou_mv``).

2. Any observation model (e.g. Poisson counts). The latent is unobserved and the
   marginal likelihood integrates it out through the observation model via the
   O(n p^3) multivariate latent tree-Laplace (``scphytr.inference.tree_laplace_mv``).
   ``fit_mv_latent`` then estimates ``(alpha, theta, K)`` by maximizing that
    marginal -- the evolutionary correlations are recovered even though the latent
    character is only seen through counts. The observation model (e.g.
    ``PoissonObservation``) is decoupled from the latent model: it is conditionally
    independent across genes, while the correlation lives in the latent ``K``.

On top of either fit it exposes the README estimators ``estimate_rate``,
``estimate_correlation`` and ``estimate_optima``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..utils.pruning import bm_pruning_logpdf, ou_pruning_logpdf, _ou_branch
from ..inference.tree_laplace_mv import mv_tree_laplace_marginal


def _set_traits(tree, trait_table):
    """Attach every column of ``trait_table`` (indexed by leaf name) to the tree."""
    tree.set_trait_values({name: row.to_dict() for name, row in trait_table.iterrows()})


def cov_to_corr(K):
    """Correlation matrix of a covariance matrix ``K``."""
    d = np.sqrt(np.diag(K))
    R = K / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


@dataclass
class FittedMVModel:
    """A fitted multivariate BM/OU model and its evolutionary parameters."""
    name: str
    trait_names: list
    loglik: float
    n_params: int
    n_obs: int
    mu: np.ndarray              # root/ancestral mean (BM) -- p vector
    K: np.ndarray               # evolutionary rate (diffusion) matrix -- p x p
    alpha: float = None         # OU mean-reversion (None for BM)
    theta: np.ndarray = None    # OU optimum per trait -- p vector (None for BM)
    extra: dict = field(default_factory=dict)

    def aic(self):
        return 2.0 * self.n_params - 2.0 * self.loglik

    def bic(self):
        return self.n_params * np.log(self.n_obs) - 2.0 * self.loglik

    def rates(self):
        """Per-trait evolutionary rate (diagonal of ``K``) as a Series."""
        return pd.Series(np.diag(self.K), index=self.trait_names, name="rate")

    def correlation(self):
        """Evolutionary correlation matrix as a DataFrame."""
        return pd.DataFrame(cov_to_corr(self.K), index=self.trait_names, columns=self.trait_names)

    def covariance(self):
        """Evolutionary covariance (rate) matrix ``K`` as a DataFrame."""
        return pd.DataFrame(self.K, index=self.trait_names, columns=self.trait_names)

    def optima(self):
        """OU optimum per trait as a Series (BM has no optimum)."""
        if self.theta is None:
            return None
        return pd.Series(self.theta, index=self.trait_names, name="optimum")


def _collect_contrasts(tree, alpha, thetas, regime_of, root_value):
    """Standardized phylogenetic contrasts (each ~ N(0, K)) over the whole tree.

    Mirrors the BM/OU pruning recursion but, instead of accumulating the
    likelihood, returns the matrix ``U`` of standardized contrasts (n x p,
    including the root contrast) plus the root belief ``(m_root, v_root)``.
    ``alpha is None`` (or <= 0) selects Brownian motion.
    """
    trait_names = tree.get_trait_names()
    is_bm = alpha is None or alpha <= 0
    contrasts = []

    def belief(node):
        if node.is_leaf():
            return np.array([node.trait[t] for t in trait_names], dtype=float), 0.0
        m_acc, v_acc = message(node.children[0])
        for child in node.children[1:]:
            m_c, v_c = message(child)
            w = v_acc + v_c
            contrasts.append((m_acc - m_c) / np.sqrt(w))
            v_new = 1.0 / (1.0 / v_acc + 1.0 / v_c)
            m_new = v_new * (m_acc / v_acc + m_c / v_c)
            m_acc, v_acc = m_new, v_new
        return m_acc, v_acc

    def message(node):
        m_b, v_b = belief(node)
        if is_bm:
            return m_b, v_b + node.dist
        phi, v = _ou_branch(alpha, node.dist)
        c = (1.0 - phi) * thetas[regime_of(node)]
        return (m_b - c) / phi, (v_b + v) / (phi * phi)

    m_root, v_root = belief(tree.root)
    if is_bm:
        v_root = v_root + tree.root.dist
        a = m_root if root_value is None else np.asarray(root_value, dtype=float).ravel()
        contrasts.append((m_root - a) / np.sqrt(v_root))
    else:
        phi_r, v_r = _ou_branch(alpha, tree.root.dist)
        a = thetas[regime_of(tree.root)] if root_value is None else np.asarray(root_value, dtype=float).ravel()
        mean_root = phi_r * a + (1.0 - phi_r) * thetas[regime_of(tree.root)]
        contrasts.append((m_root - mean_root) / np.sqrt(v_root + v_r))

    return np.array(contrasts), m_root, v_root


def _n_cov_params(p):
    """Free parameters in a symmetric p x p covariance matrix."""
    return p * (p + 1) // 2


def fit_bm_mv(tree, trait_table):
    """Fit multivariate Brownian motion; estimate root mean and full rate matrix K.

    Returns a :class:`FittedMVModel`. ``K`` is the maximum-likelihood diffusion
    matrix (sample covariance of contrasts); its off-diagonals are the evolutionary
    covariances between traits and ``cov_to_corr(K)`` the evolutionary correlations.
    """
    _set_traits(tree, trait_table)
    trait_names = list(trait_table.columns)
    p = len(trait_names)
    n = len(tree.root.get_leaves())

    U, m_root, _ = _collect_contrasts(tree, None, None, None, None)
    K = (U.T @ U) / n                      # ML (root contrast is zero at mu = m_root)
    mu = m_root
    loglik = bm_pruning_logpdf(tree, mu, K)
    n_params = p + _n_cov_params(p)
    return FittedMVModel("BM", trait_names, loglik, n_params, n, mu=mu, K=K)


def _tree_height(tree):
    return float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)


def fit_ou_mv(tree, trait_table, alpha_inits=(0.25, 1.0, 4.0), seed=0):
    """Fit multivariate single-optimum OU: shared alpha, per-trait optimum, full K.

    ``alpha`` and the optimum vector ``theta`` are found by low-dimensional
    optimization; given them, ``K`` is profiled out in closed form as the contrast
    sample covariance. The ancestral root is tied to ``theta`` (resolving the
    root/optimum identifiability), so the marginal mean is constant at ``theta``.
    """
    _set_traits(tree, trait_table)
    trait_names = list(trait_table.columns)
    p = len(trait_names)
    n = len(tree.root.get_leaves())
    alpha_max = 30.0 / max(_tree_height(tree), 1e-12)
    Y = trait_table.values.astype(float)
    mean0 = Y.mean(axis=0)

    regime_of = lambda node: 0

    def profile(alpha, theta):
        thetas = theta.reshape(1, p)
        U, _, _ = _collect_contrasts(tree, alpha, thetas, regime_of, theta)
        K = (U.T @ U) / n
        ll = ou_pruning_logpdf(tree, alpha, thetas, K, root_value=theta)
        return K, ll

    def nll(x):
        alpha = float(np.clip(np.exp(x[0]), 1e-4, alpha_max))
        theta = x[1:]
        try:
            _, ll = profile(alpha, theta)
        except (np.linalg.LinAlgError, ValueError):
            return 1e18
        return -ll if np.isfinite(ll) else 1e18

    best = None
    for a0 in alpha_inits:
        x0 = np.concatenate([[np.log(min(a0, alpha_max))], mean0])
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000 + 1000 * p})
        if best is None or res.fun < best.fun:
            best = res

    alpha = float(np.clip(np.exp(best.x[0]), 1e-4, alpha_max))
    theta = best.x[1:]
    K, loglik = profile(alpha, theta)
    n_params = 1 + p + _n_cov_params(p)
    return FittedMVModel("OU", trait_names, loglik, n_params, n,
                         mu=theta.copy(), K=K, alpha=alpha, theta=theta)


def fit_mv(tree, trait_table, model="BM", **kwargs):
    """Fit a multivariate model to directly-observed traits: ``model`` in {"BM","OU"}."""
    if model == "BM":
        return fit_bm_mv(tree, trait_table)
    if model == "OU":
        return fit_ou_mv(tree, trait_table, **kwargs)
    raise ValueError(f"Unknown model '{model}' (expected 'BM' or 'OU').")


def _chol_pack(K):
    """Unconstrained vector (lower-tri with log-diagonal) for a PD matrix ``K``."""
    p = K.shape[0]
    L = np.linalg.cholesky(K)
    tri = np.tril_indices(p)
    v = L[tri].copy()
    v[tri[0] == tri[1]] = np.log(v[tri[0] == tri[1]])
    return v


def _chol_unpack(v, p):
    """PD matrix ``K = L L^T`` from the unconstrained vector produced by _chol_pack."""
    L = np.zeros((p, p))
    tri = np.tril_indices(p)
    L[tri] = v
    di = np.diag_indices(p)
    L[di] = np.exp(L[di])
    return L @ L.T


def fit_mv_latent(tree, obs, model="BM", trait_names=None, regimes=None,
                  alpha_inits=(0.25, 1.0, 4.0), seed=0, restarts=1):
    """Estimate latent multivariate BM/OU parameters under *any* observation model.

    Maximizes the O(n p^3) multivariate latent tree-Laplace marginal over the
    latent-model parameters: the diffusion matrix ``K`` (full, PD via a Cholesky
    parameterization), the optimum ``theta`` (per regime for OU), and -- for OU --
    the mean-reversion ``alpha``. The observation model ``obs`` (e.g.
    ``PoissonObservation``) only enters through its leaf likelihood, so any
    non-conjugate model is supported.

    Returns a :class:`FittedMVModel`; ``K``'s off-diagonals are the evolutionary
    covariances and ``correlation()`` the evolutionary correlations between genes.
    """
    is_ou = model == "OU"
    init = obs.mode_init()                     # (n, p)
    p = init.shape[1]
    if trait_names is None:
        trait_names = list(range(p))
    n_reg = 1 if regimes is None else len(set(regimes.values()))
    root_regime = None if regimes is None else regimes[tree.root]
    alpha_max = 30.0 / max(_tree_height(tree), 1e-12)

    theta0 = init.mean(axis=0)
    K0 = (np.cov(init, rowvar=False) if p > 1 else np.array([[max(np.var(init), 1e-2)]]))
    K0 = np.atleast_2d(K0) + 1e-3 * np.eye(p)
    v0 = _chol_pack(K0)
    n_theta = n_reg * p
    n_L = p * (p + 1) // 2

    def split(x):
        off = 0
        alpha = 0.0
        if is_ou:
            alpha = float(np.clip(np.exp(x[0]), 1e-4, alpha_max))
            off = 1
        thetas = x[off:off + n_theta].reshape(n_reg, p)
        K = _chol_unpack(x[off + n_theta:off + n_theta + n_L], p)
        return alpha, thetas, K

    def nll(x):
        alpha, thetas, K = split(x)
        theta_arg = thetas if regimes is not None else thetas[0]
        rv = thetas[root_regime] if regimes is not None else thetas[0]
        try:
            ll = mv_tree_laplace_marginal(tree, obs, alpha, theta_arg, K,
                                          regimes=regimes, root_value=rv)
        except (np.linalg.LinAlgError, ValueError):
            return 1e18
        return -ll if np.isfinite(ll) else 1e18

    rng = np.random.default_rng(seed)
    a_starts = alpha_inits if is_ou else (None,)
    best = None
    for a0 in a_starts:
        for r in range(restarts):
            theta_start = np.tile(theta0, n_reg) + (0 if r == 0 else 0.3 * rng.standard_normal(n_theta))
            head = [np.log(min(a0, alpha_max))] if is_ou else []
            x0 = np.concatenate([np.array(head), theta_start, v0])
            res = minimize(nll, x0, method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-7,
                                    "maxiter": 3000 + 1500 * (n_theta + n_L)})
            if best is None or res.fun < best.fun:
                best = res

    alpha, thetas, K = split(best.x)
    n_params = (1 if is_ou else 0) + n_theta + n_L
    n = len(tree.root.get_leaves())
    name = "OU" if is_ou else "BM"
    theta_out = thetas if regimes is not None else thetas[0]
    return FittedMVModel(name, list(trait_names), loglik=-best.fun, n_params=n_params,
                         n_obs=n, mu=(thetas[0].copy() if is_ou else thetas[0].copy()),
                         K=K, alpha=(alpha if is_ou else None),
                         theta=(theta_out if is_ou else None),
                         extra={"n_regimes": n_reg})


# ---------------------------------------------------------------------------
# README-facing estimators (tree + trait table in, tidy pandas out).
# ---------------------------------------------------------------------------

def _fit_either(tree, trait_table=None, obs=None, model="BM", trait_names=None,
                method="em", **kwargs):
    """Dispatch to the conjugate (trait_table) or latent (obs) fitter.

    For directly-observed traits (``trait_table``) the conjugate closed-form
    contrast MLE is used. For a latent character seen through an observation model
    (``obs``) the parameters are optimized over the latent tree-Laplace marginal:
    ``method="em"`` (default) runs the Laplace-EM with JAX-gradient M-steps;
    ``method="direct"`` runs the gradient-free optimizer.
    """
    if (trait_table is None) == (obs is None):
        raise ValueError("Provide exactly one of `trait_table` (directly observed) "
                         "or `obs` (latent through an observation model).")
    if obs is not None:
        names = trait_names if trait_names is not None else (
            list(trait_table.columns) if trait_table is not None else None)
        if method == "em":
            from .em import fit_mv_em          # lazy import (JAX) to avoid a cycle
            return fit_mv_em(tree, obs, model=model, trait_names=names, **kwargs)
        if method == "direct":
            return fit_mv_latent(tree, obs, model=model, trait_names=names, **kwargs)
        raise ValueError(f"Unknown method '{method}' (expected 'em' or 'direct').")
    return fit_mv(tree, trait_table, model=model, **kwargs)


def estimate_rate(tree, trait_table=None, obs=None, model="BM", **kwargs):
    """Evolutionary rate matrix ``K`` (DataFrame) for a set of traits/genes.

    The diagonal is the per-gene rate; off-diagonals are evolutionary covariances.
    Pass ``trait_table`` for directly-observed traits, or ``obs`` (e.g.
    ``PoissonObservation``) to estimate the *latent* rate matrix under any
    observation model. For the latent path the parameters are fit by Laplace-EM
    (``method="em"``, default; ``"direct"`` for the gradient-free optimizer). Use
    ``estimate_correlation`` for the normalized version.
    """
    return _fit_either(tree, trait_table, obs, model=model, **kwargs).covariance()


def estimate_correlation(tree, trait_table=None, obs=None, model="BM", **kwargs):
    """Evolutionary correlation matrix between traits/genes (DataFrame).

    Works on directly-observed traits (``trait_table``) or on a latent character
    observed through any model (``obs``), e.g. correlated gene evolution from
    Poisson counts. The latent path is fit by Laplace-EM by default
    (``method="em"``; pass ``method="direct"`` for the gradient-free optimizer).
    """
    return _fit_either(tree, trait_table, obs, model=model, **kwargs).correlation()


def estimate_optima(tree, trait_table=None, obs=None, **kwargs):
    """OU evolutionary optimum per trait/gene (Series), from a multivariate OU fit.

    Latent (``obs``) fits use Laplace-EM by default (``method="em"``).
    """
    return _fit_either(tree, trait_table, obs, model="OU", **kwargs).optima()
