"""Laplace-approximate marginal likelihood for latent Gaussian tree models.

The trait evolves on the tree as a latent Gaussian process (BM/OU prior, mean
``m`` and covariance ``Sigma``), observed through a per-leaf likelihood that is not
conjugate (e.g. Poisson counts). The marginal likelihood

    p(y | hyperparams) = ∫ p(y | z) N(z; m, Sigma) dz

is approximated by Laplace's method around the posterior mode. For a log-concave
likelihood the mode is unique and Newton's method converges; we use the numerically
stable formulation of Rasmussen & Williams (GPML, Algorithm 3.1), generalized to a
non-zero prior mean.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

import jax.numpy as jnp
from jax.scipy.special import gammaln as jgammaln


class PoissonObservation:
    """Poisson counts with log-link and fixed offsets: Y ~ Poisson(S e^z).

    This is purely an observation model and says nothing about the latent process:
    given the latent log-rate ``z`` it is conditionally independent across all
    entries (leaves, and genes if present). It works unchanged whether the latent
    is a scalar per leaf (``(n,)``, univariate tree-Laplace) or a vector per leaf
    (``(n, p)``, multivariate latent tree-Laplace) -- any cross-gene correlation
    lives entirely in the latent BM/OU model, not here.

    Parameters
    ----------
    counts : array, shape (n,) or (n, p)
        Aggregated counts per leaf (and per gene), summed over a species' cells.
    offsets : array
        Size factors S > 0, broadcastable to ``counts`` (e.g. ``(n,)`` offsets are
        shared across genes when ``counts`` is ``(n, p)``).
    """

    def __init__(self, counts, offsets):
        self.y = np.asarray(counts, dtype=float)
        S = np.asarray(offsets, dtype=float)
        if S.ndim == 1 and self.y.ndim == 2:
            S = S[:, None]
        self.S = S * np.ones_like(self.y)
        if np.any(self.S <= 0):
            raise ValueError("Poisson offsets S must be positive.")

    def loglik(self, f):
        rate = self.S * np.exp(f)
        return float(np.sum(self.y * f - rate - gammaln(self.y + 1.0)))

    def grad(self, f):
        return self.y - self.S * np.exp(f)

    def neg_hess_diag(self, f):
        # W = -d^2/df^2 log p = S e^{f} > 0 (diagonal: entries are independent given z)
        return self.S * np.exp(f)

    def loglik_jax(self, f):
        """JAX (differentiable) mirror of :meth:`loglik`, for gradient-based engines."""
        rate = jnp.asarray(self.S) * jnp.exp(f)
        return jnp.sum(jnp.asarray(self.y) * f - rate - jgammaln(jnp.asarray(self.y) + 1.0))

    def mode_init(self):
        # Data-driven start near the MLE of each independent rate.
        return np.log((self.y + 0.5) / (self.S + 1e-9))


class GaussianObservation:
    """Gaussian observation y ~ N(z, tau). Conjugate, so Laplace is exact.

    Like :class:`PoissonObservation`, this is purely an observation model and is
    independent across entries given the latent. It works for scalar-per-leaf
    ``(n,)`` or vector-per-leaf ``(n, p)`` latents. Useful as an exact oracle and
    as a measurement-error model for directly-observed traits (small ``tau``).

    Parameters
    ----------
    y : array, shape (n,) or (n, p) -- observed values.
    tau : observation variance, broadcastable to ``y``.
    """

    def __init__(self, y, tau):
        self.y_obs = np.asarray(y, dtype=float)
        self.tau = np.asarray(tau, dtype=float) * np.ones_like(self.y_obs)

    def loglik(self, f):
        return float(np.sum(-0.5 * ((self.y_obs - f) ** 2 / self.tau
                                    + np.log(2.0 * np.pi * self.tau))))

    def grad(self, f):
        return (self.y_obs - f) / self.tau

    def neg_hess_diag(self, f):
        return np.ones_like(f) / self.tau

    def loglik_jax(self, f):
        """JAX (differentiable) mirror of :meth:`loglik`, for gradient-based engines."""
        y = jnp.asarray(self.y_obs)
        tau = jnp.asarray(self.tau)
        return jnp.sum(-0.5 * ((y - f) ** 2 / tau + jnp.log(2.0 * jnp.pi * tau)))

    def mode_init(self):
        return self.y_obs.copy()


class MultiCellPoissonObservation:
    """Multiple cell counts per tree leaf (e.g. cells within a subclone).

    The leaves of the tree are *subclones* (or any aggregate); each leaf carries
    several cells. Every cell ``i`` assigned to leaf ``leaf_index[i]`` is observed
    as ``Y_i ~ Poisson(S_i e^{z_leaf + eps_i})`` and the latent lives at the leaf,
    so this exposes the standard observation interface on the **leaf**-latent
    ``F`` of shape ``(n_leaves, p)`` -- summing/scattering each leaf's cells.

    Two regimes (the modeling point of subclone-resolution lineages):

    - ``dispersion is None`` -- *pure Poisson* (``eps_i = 0``): cells are exact
      replicates of the subclone latent. The sufficient statistics collapse, so
      this is identical (in grad/curvature) to :class:`PoissonObservation` on the
      per-leaf **summed** counts and offsets -- i.e. summed-count pseudobulk.
    - ``dispersion = r`` (per-gene NB size, scalar or ``(p,)``) -- a within-leaf
      **Gamma-Poisson** overdispersion that does *not* collapse: it models
      genuine intra-subclone heterogeneity (plasticity) beyond shot noise, the
      count-level analogue of the EVE within-population variance. ``r -> inf``
      recovers pure Poisson. (Gamma-Poisson is the closed-form stand-in for a
      per-cell log-normal ``eps_i ~ N(0, D)``; both add a per-cell random effect.)

    Parameters
    ----------
    counts : array (n_cells, p)        -- per-cell counts.
    offsets : array (n_cells,) or (n_cells, p) -- per-cell size factors S_i > 0.
    leaf_index : int array (n_cells,)  -- leaf each cell belongs to, in [0, n_leaves).
    n_leaves : int                     -- number of tree leaves.
    dispersion : None | float | array (p,) -- NB size r per gene; None => Poisson.
    """

    def __init__(self, counts, offsets, leaf_index, n_leaves, dispersion=None,
                 univariate=False):
        self.y = np.asarray(counts, dtype=float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        n_cells, p = self.y.shape
        # ``univariate`` makes mode_init return a 1-D (n_leaves,) array so the
        # fast scalar tree-Laplace path (``tree_laplace.py``) can be used for a
        # single gene; loglik/grad/neg_hess already adapt to 1-D vs 2-D input.
        self.univariate = bool(univariate) and p == 1
        S = np.asarray(offsets, dtype=float)
        if S.ndim == 1:
            S = S[:, None]
        self.S = S * np.ones_like(self.y)
        if np.any(self.S <= 0):
            raise ValueError("Poisson offsets S must be positive.")
        self.idx = np.asarray(leaf_index, dtype=int)
        if self.idx.shape[0] != n_cells:
            raise ValueError("leaf_index must have one entry per cell.")
        self.n_leaves = int(n_leaves)
        self.p = p
        self.r = None if dispersion is None else np.broadcast_to(
            np.asarray(dispersion, dtype=float), (p,)).copy()
        # Per-leaf summed sufficient statistics (for mode_init and the pure case).
        self.Ytot = np.zeros((self.n_leaves, p))
        self.Stot = np.zeros((self.n_leaves, p))
        np.add.at(self.Ytot, self.idx, self.y)
        np.add.at(self.Stot, self.idx, self.S)

    def _mu(self, f):
        # per-cell mean: mu_i = S_i exp(f_{leaf(i)}). ``f`` may be (n_leaves,)
        # (univariate/scalar tree-Laplace path) or (n_leaves, p) (multivariate).
        f2 = f[:, None] if f.ndim == 1 else f
        return self.S * np.exp(f2[self.idx])

    def loglik(self, f):
        mu = self._mu(f)
        y = self.y
        if self.r is None:
            return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))
        r = self.r[None, :]
        return float(np.sum(
            gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))

    def grad(self, f):
        mu = self._mu(f)
        if self.r is None:
            g_cell = self.y - mu                       # d/df, f shared within leaf
        else:
            r = self.r[None, :]
            g_cell = self.y - (self.y + r) * mu / (r + mu)
        out = np.zeros((self.n_leaves, self.p))
        np.add.at(out, self.idx, g_cell)
        return out[:, 0] if f.ndim == 1 else out

    def neg_hess_diag(self, f):
        mu = self._mu(f)
        if self.r is None:
            w_cell = mu                                # -d^2/df^2 = S e^f
        else:
            r = self.r[None, :]
            w_cell = (self.y + r) * mu * r / (r + mu) ** 2
        out = np.zeros((self.n_leaves, self.p))
        np.add.at(out, self.idx, w_cell)
        return out[:, 0] if f.ndim == 1 else out

    def mode_init(self):
        m = np.log((self.Ytot + 0.5) / (self.Stot + 1e-9))
        return m[:, 0] if self.univariate else m

    def update_dispersion(self, f, bounds=(1e-2, 1e6)):
        """Re-estimate the per-gene NB size ``r`` at fixed leaf latent ``f``.

        This is the *observation* M-step of the EM: with the subclone latents
        held at the current posterior mode ``f`` (so the per-cell means
        ``mu = S_i e^{f_leaf}`` are fixed), maximize the within-leaf Gamma-Poisson
        log-likelihood over the dispersion ``r_g`` for each gene -- the standard
        NB-dispersion MLE. Mutates and returns ``self.r``. The estimated ``r_g``
        is the within-subclone (plastic) overdispersion, complementary to the
        heritable diffusion ``K`` estimated by the latent M-step.
        """
        from scipy.optimize import minimize_scalar
        mu = self._mu(f)
        r_new = np.empty(self.p)
        lo, hi = np.log(bounds[0]), np.log(bounds[1])
        for g in range(self.p):
            y = self.y[:, g]; m = mu[:, g]

            def nll(logr, y=y, m=m):
                r = np.exp(logr)
                return -float(np.sum(gammaln(y + r) - gammaln(r)
                                     + r * np.log(r / (r + m)) + y * np.log(m / (r + m))))

            res = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
            r_new[g] = np.exp(res.x)
        self.r = r_new
        return r_new


def laplace_posterior(obs, mean, Sigma, max_iter=100, tol=1e-8, f_clip=40.0):
    """Laplace approximation: posterior mode, marginal log-likelihood, covariance.

    Parameters
    ----------
    obs : object with loglik(f), grad(f), neg_hess_diag(f), mode_init()
    mean : array (n,)   -- prior mean vector
    Sigma : array (n, n) -- prior covariance (PD)

    Returns
    -------
    dict with keys:
        'logZ' : Laplace log marginal likelihood,
        'mode' : posterior mode f-hat,
        'cov'  : Gaussian posterior covariance (K^{-1} + W)^{-1}.
    """
    mean = np.asarray(mean, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mean.shape[0]
    eye = np.eye(n)

    def objective(f):
        a = np.linalg.solve(Sigma, f - mean)
        return obs.loglik(f) - 0.5 * float((f - mean) @ a)

    # Initialize between the data-driven mode and the prior mean for stability.
    f = np.clip(0.5 * (obs.mode_init() + mean), mean - f_clip, mean + f_clip)
    psi = objective(f)

    for _ in range(max_iter):
        W = obs.neg_hess_diag(f)
        sW = np.sqrt(W)
        B = eye + sW[:, None] * Sigma * sW[None, :]
        cho = cho_factor(B, lower=True)
        b = W * (f - mean) + obs.grad(f)
        a = b - sW * cho_solve(cho, sW * (Sigma @ b))
        step = mean + Sigma @ a - f  # Newton direction

        # Backtracking line search to guarantee monotone increase of the objective.
        t = 1.0
        psi_try = psi
        for _ in range(40):
            f_try = np.clip(f + t * step, mean - f_clip, mean + f_clip)
            psi_try = objective(f_try)
            if psi_try >= psi:
                break
            t *= 0.5
        converged = abs(psi_try - psi) < tol
        f, psi = f_try, psi_try
        if converged:
            break

    # Final quantities at the mode.
    W = obs.neg_hess_diag(f)
    sW = np.sqrt(W)
    B = eye + sW[:, None] * Sigma * sW[None, :]
    cho = cho_factor(B, lower=True)
    a = np.linalg.solve(Sigma, f - mean)
    log_det_B = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
    logZ = obs.loglik(f) - 0.5 * float((f - mean) @ a) - 0.5 * log_det_B

    # Posterior covariance (K^{-1}+W)^{-1} = K - K sW B^{-1} sW K  (R&W 3.27).
    V = cho_solve(cho, (sW[:, None] * Sigma))  # B^{-1} (sW K)
    cov = Sigma - (sW[:, None] * Sigma).T @ V
    return {"logZ": logZ, "mode": f, "cov": cov}


def laplace_marginal_loglik(obs, mean, Sigma, **kwargs):
    """Laplace approximation to log p(y | hyperparams). See ``laplace_posterior``."""
    return laplace_posterior(obs, mean, Sigma, **kwargs)["logZ"]
