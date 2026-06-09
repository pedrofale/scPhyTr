"""Pluggable E-step inference engines for the latent multivariate tree model.

The model is a *latent Gaussian tree*: a BM/OU prior with sparse precision
``Q = A ⊗ K^{-1}`` (handled by :class:`~scphytr.inference.tree_laplace_mv._MVTreeModel`)
observed at the leaves through a (generally non-conjugate) decoder ``p(Y | Z)``.
The EM E-step needs the posterior over the latent node-vectors, summarized as the
first/second moments the M-step consumes. Every engine here returns the *same*
contract as ``mv_laplace_estep``::

    {"M": _MVTreeModel, "Z": (N, p) posterior mean,
     "Sigma": (N, p, p) node covariances, "cross": (N, p, p) parent-child cov}

so :func:`scphytr.tools.em.fit_mv_em` can swap engines without touching the
JAX-gradient M-step. Only the way ``Z``/``Sigma``/``cross`` are computed differs:

* :class:`LaplaceEStep`            -- posterior mode + RTS smoother (default; exact for Gaussian obs).
* :class:`ImportanceSamplingEStep` -- self-normalized IS with the Laplace Gaussian as proposal.
* :class:`MCMCEStep`               -- blackjax NUTS over the latent (gradient-based).

The latter two also expose diagnostics (``ess``, ``logZ``, acceptance) but those
are ignored by the M-step.
"""

import numpy as np
from scipy.special import logsumexp

from .tree_laplace_mv import _MVTreeModel, _newton_mode, mv_laplace_estep


# ---------------------------------------------------------------------------
# Shared helpers (vectorized over samples).
# ---------------------------------------------------------------------------
def _edge_arrays(M):
    """Static per-node arrays for vectorized prior quadratics (postorder order)."""
    phi_eff = np.where(M.is_root, 0.0, M.phi)
    parent_eff = np.where(M.parent < 0, 0, M.parent)
    invV_eff = M.invV * M.free
    cmean = M.c.copy()
    cmean[M.is_root] = M.mu0[M.is_root]            # root: d = Z_root - mu0_root
    return phi_eff, parent_eff, invV_eff, cmean


def _prior_quad_batch(M, Z, centered=False):
    """Per-sample (Z - m)^T Q (Z - m) (or Z^T Q Z if ``centered``).

    ``Z`` has shape (S, N, p); returns shape (S,).
    """
    phi_eff, parent_eff, invV_eff, cmean = _edge_arrays(M)
    d = Z - phi_eff[None, :, None] * Z[:, parent_eff, :]
    if not centered:
        d = d - cmean[None, :, :]
    q = np.einsum("sip,pq,siq->si", d, M.P, d)     # quadratic per node
    return q @ invV_eff                            # weight by 1/V and sum over nodes


def _weighted_moments(M, samples, weights):
    """Posterior mean and node / parent-child covariance blocks from weighted samples.

    ``samples`` is (S, N, p); ``weights`` sums to 1. Returns (Z, Sigma, cross).
    """
    Z = np.einsum("s,snp->np", weights, samples)
    dev = samples - Z[None]
    Sigma = np.einsum("s,snp,snq->npq", weights, dev, dev)
    par = M.solve_parent
    par_eff = np.where(par < 0, 0, par)
    dev_pa = dev[:, par_eff, :]
    cross = np.einsum("s,snp,snq->npq", weights, dev, dev_pa)
    cross[par < 0] = 0.0
    return Z, Sigma, cross


# ---------------------------------------------------------------------------
# Engines.
# ---------------------------------------------------------------------------
class LaplaceEStep:
    """Laplace E-step: posterior mode + RTS smoother covariances (the default)."""

    def __init__(self, max_iter=100, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def run(self, tree, obs, alpha, theta, K, regimes=None, root_value=None):
        return mv_laplace_estep(tree, obs, alpha, theta, K, regimes=regimes,
                                root_value=root_value, max_iter=self.max_iter, tol=self.tol)


class ImportanceSamplingEStep:
    """Self-normalized importance sampling with the Laplace Gaussian as proposal.

    The proposal is ``q = N(mode, (Q + W)^{-1})`` (the Laplace posterior), sampled
    exactly via the tree simulation smoother. Importance weights correct ``q``
    toward the true posterior ``p(Z | Y) ∝ p(Y | Z) N(Z; m, Q^{-1})``; posterior
    moments are the weighted sample moments. Also returns the effective sample
    size ``ess`` and a near-unbiased marginal estimate ``logZ`` (a check on the
    Laplace marginal).
    """

    def __init__(self, n_samples=4000, seed=0, max_iter=100, tol=1e-8):
        self.n_samples = n_samples
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def run(self, tree, obs, alpha, theta, K, regimes=None, root_value=None):
        M = _MVTreeModel(tree, alpha, theta, K, regimes=regimes, root_value=root_value)
        if not M.free[M.root_idx]:
            raise ValueError("E-step requires a free root (positive root branch length).")
        mode = _newton_mode(M, obs, max_iter=self.max_iter, tol=self.tol)

        leaf = M.leaf_node_idx
        Wdiag = np.zeros((M.N, M.p))
        Wdiag[leaf] = obs.neg_hess_diag(mode[leaf])

        rng = np.random.default_rng(self.seed)
        Zs = M.sample_gaussian(Wdiag, mode, self.n_samples, rng)     # (S, N, p)

        # log weights (up to a constant that cancels in self-normalization):
        #   log p(Y, Z) - log q(Z)
        #     = loglik(F) - 0.5 (Z-m)^T Q (Z-m) + 0.5 (Z-mode)^T (Q+W) (Z-mode) + const.
        delta = Zs - mode[None]
        prior_q = _prior_quad_batch(M, Zs, centered=False)
        prop_q = (_prior_quad_batch(M, delta, centered=True)
                  + np.einsum("snp,np->s", delta ** 2, Wdiag))
        loglik = np.array([obs.loglik(Zs[s, leaf]) for s in range(self.n_samples)])
        logw = loglik - 0.5 * prior_q + 0.5 * prop_q

        # Absolute marginal: add the constant 0.5(log|Q| - log|Q+W|); the (2π)
        # factors cancel between prior and proposal.
        log_const = 0.5 * (M.log_det_Q - M.log_det(Wdiag))
        logZ = float(logsumexp(logw) - np.log(self.n_samples) + log_const)

        wn = np.exp(logw - logsumexp(logw))
        ess = float(1.0 / np.sum(wn ** 2))
        Z, Sigma, cross = _weighted_moments(M, Zs, wn)
        return {"M": M, "Z": Z, "Sigma": Sigma, "cross": cross,
                "ess": ess, "logZ": logZ, "n_samples": self.n_samples}


class MCMCEStep:
    """NUTS (blackjax) over the latent node-vectors; moments are sample statistics.

    The log-density is ``obs.loglik_jax(F) - 0.5 (Z - m)^T Q (Z - m)`` evaluated in
    JAX, so any decoder exposing a differentiable ``loglik_jax`` is supported.
    Requires a free root. Returns the posterior moments plus diagnostics
    ``ess`` and ``accept`` (mean acceptance probability).
    """

    def __init__(self, n_samples=2000, n_warmup=1000, seed=0, max_iter=100, tol=1e-8):
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def _logdensity_fn(self, M, obs):
        import jax.numpy as jnp

        phi_eff, parent_eff, invV_eff, cmean = _edge_arrays(M)
        phi_j = jnp.asarray(phi_eff)
        par_j = jnp.asarray(parent_eff)
        invV_j = jnp.asarray(invV_eff)
        cmean_j = jnp.asarray(cmean)
        P_j = jnp.asarray(M.P)
        leaf_j = jnp.asarray(M.leaf_node_idx)
        N, p = M.N, M.p

        def logdensity(z_flat):
            Z = z_flat.reshape(N, p)
            d = Z - phi_j[:, None] * Z[par_j] - cmean_j
            quad = jnp.einsum("ip,pq,iq->i", d, P_j, d)
            prior = -0.5 * jnp.sum(invV_j * quad)
            return prior + obs.loglik_jax(Z[leaf_j])

        return logdensity, N, p

    def run(self, tree, obs, alpha, theta, K, regimes=None, root_value=None):
        import jax
        import jax.numpy as jnp
        import blackjax

        M = _MVTreeModel(tree, alpha, theta, K, regimes=regimes, root_value=root_value)
        if not M.free[M.root_idx]:
            raise ValueError("E-step requires a free root (positive root branch length).")
        mode = _newton_mode(M, obs, max_iter=self.max_iter, tol=self.tol)

        logdensity, N, p = self._logdensity_fn(M, obs)
        z0 = jnp.asarray(mode.reshape(-1))

        key = jax.random.PRNGKey(self.seed)
        warmup_key, sample_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
        (state, parameters), _ = warmup.run(warmup_key, z0, num_steps=self.n_warmup)
        kernel = blackjax.nuts(logdensity, **parameters)

        def step(carry, k):
            st, info = kernel.step(k, carry)
            return st, (st.position, info.acceptance_rate)

        keys = jax.random.split(sample_key, self.n_samples)
        _, (positions, accept) = jax.lax.scan(step, state, keys)

        samples = np.asarray(positions).reshape(self.n_samples, N, p)
        weights = np.full(self.n_samples, 1.0 / self.n_samples)
        Z, Sigma, cross = _weighted_moments(M, samples, weights)

        try:
            ess = float(np.asarray(
                blackjax.diagnostics.effective_sample_size(positions[None, ...])).mean())
        except Exception:
            ess = float("nan")
        return {"M": M, "Z": Z, "Sigma": Sigma, "cross": cross,
                "ess": ess, "accept": float(np.asarray(accept).mean()),
                "n_samples": self.n_samples}


_ENGINES = {
    "laplace": LaplaceEStep,
    "is": ImportanceSamplingEStep,
    "importance": ImportanceSamplingEStep,
    "mcmc": MCMCEStep,
    "nuts": MCMCEStep,
}


def resolve_estep(engine):
    """Map a string name or pass-through an engine instance to an E-step engine."""
    if engine is None:
        return LaplaceEStep()
    if isinstance(engine, str):
        try:
            return _ENGINES[engine.lower()]()
        except KeyError:
            raise ValueError(f"Unknown E-step engine '{engine}'. "
                             f"Choose from {sorted(set(_ENGINES))} or pass an engine instance.")
    return engine                                  # assume an engine instance with .run(...)
