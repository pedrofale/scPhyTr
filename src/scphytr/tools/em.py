"""Laplace-EM for the latent multivariate BM/OU model, with JAX-gradient M-steps.

The latent evolutionary parameters (mean-reversion ``alpha``, optimum ``theta``,
and the full diffusion matrix ``K``) are fit by Expectation-Maximization:

* E-step -- given the current parameters, the Laplace posterior over the latent
  node-vectors is computed (mode + covariance blocks) by the O(n p^3) block-tree
  smoother in ``scphytr.inference.tree_laplace_mv``. From it we read the expected
  per-edge sufficient statistics E[Δz Δzᵀ]. The observation model enters *only*
  here, so any non-conjugate likelihood (Poisson counts, ...) is supported.

* M-step -- the expected complete-data log-likelihood (the Gaussian "Q-function")
  is a smooth function of (alpha, theta, K). We maximize it with JAX automatic
  differentiation (BFGS on the exact gradient), parameterizing ``K = L Lᵀ`` via a
  Cholesky factor so it stays positive-definite. This is where JAX is used: the
  M-step objective and its gradient are evaluated by ``jax.grad``.

EM monotonically improves the (Laplace) marginal; we monitor it for convergence.
The root must be a free latent (positive root branch length).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize

from ..inference.tree_laplace_mv import mv_tree_laplace_marginal
from ..inference.estep import resolve_estep
from .estimation import FittedMVModel

jax.config.update("jax_enable_x64", True)


def _make_mstep_objective(moments, p, n_reg, is_ou):
    """Build the (negative) expected complete-data log-likelihood and param splitter.

    ``moments`` holds the E-step posterior statistics as jnp arrays:
    m (N,p), Sig (N,p,p), m_pa (N,p), Sig_pa (N,p,p), cross (N,p,p),
    t (N,), regime (N,), root_mask (N,).
    """
    m = moments["m"]; Sig = moments["Sig"]
    m_pa = moments["m_pa"]; Sig_pa = moments["Sig_pa"]; cross = moments["cross"]
    t = moments["t"]; regime = moments["regime"]; root_mask = moments["root_mask"]

    tri = np.tril_indices(p)
    n_L = p * (p + 1) // 2
    diag_in_tri = np.diag_indices(p)

    def unpack_L(v):
        L = jnp.zeros((p, p)).at[tri].set(v)
        return L.at[diag_in_tri].set(jnp.exp(L[diag_in_tri]))

    def split(x):
        off = 0
        alpha = 0.0
        if is_ou:
            alpha = jnp.exp(x[0])
            off = 1
        theta = x[off:off + n_reg * p].reshape(n_reg, p)
        L = unpack_L(x[off + n_reg * p:off + n_reg * p + n_L])
        return alpha, theta, L

    def neg_Q(x):
        alpha, theta, L = split(x)
        K = L @ L.T
        P = jnp.linalg.inv(K)
        logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        if is_ou:
            phi_real = jnp.exp(-alpha * t)
            v = -jnp.expm1(-2.0 * alpha * t) / (2.0 * alpha)
        else:
            phi_real = jnp.ones_like(t)
            v = t

        theta_node = theta[regime]                                  # (N, p)
        # Root edge: optimum-tied ancestor, so the mean is theta and phi acts as 0.
        c = jnp.where(root_mask[:, None], theta_node, (1.0 - phi_real)[:, None] * theta_node)
        phi_m = jnp.where(root_mask, 0.0, phi_real)

        Ezz = Sig + m[:, :, None] * m[:, None, :]
        Ezpa = cross + m[:, :, None] * m_pa[:, None, :]
        Epapa = Sig_pa + m_pa[:, :, None] * m_pa[:, None, :]
        quad = (Ezz - phi_m[:, None, None] * (Ezpa + jnp.swapaxes(Ezpa, 1, 2))
                + (phi_m ** 2)[:, None, None] * Epapa)
        dvec = m - phi_m[:, None] * m_pa
        Mblk = (quad - dvec[:, :, None] * c[:, None, :]
                - c[:, :, None] * dvec[:, None, :] + c[:, :, None] * c[:, None, :])
        trPM = jnp.sum(P[None] * Mblk, axis=(1, 2))

        Q = jnp.sum(-0.5 * (p * jnp.log(v) + logdetK + trPM / v))
        return -Q

    return neg_Q, split


def _pack_L(K, p):
    tri = np.tril_indices(p)
    L = np.linalg.cholesky(K)
    v = L[tri].copy()
    v[tri[0] == tri[1]] = np.log(v[tri[0] == tri[1]])
    return v


def fit_mv_em(tree, obs, model="BM", trait_names=None, regimes=None,
              max_em=50, tol=1e-4, alpha0=1.0, mstep_maxiter=200, verbose=False,
              estep="laplace"):
    """Fit latent multivariate BM/OU under any observation model via Laplace-EM.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    obs : observation model exposing loglik/grad/neg_hess_diag/mode_init on (n, p)
        leaf-latent arrays (e.g. ``PoissonObservation``).
    model : {"BM", "OU"}
    regimes : dict[node -> int], optional (per-regime optima for OU).
    estep : str or E-step engine, optional
        Which engine computes the E-step posterior moments. One of ``"laplace"``
        (default), ``"is"`` (importance sampling), ``"mcmc"`` (blackjax NUTS), or
        an engine instance with a ``run(...)`` method. Only the E-step changes;
        the JAX-gradient M-step is identical across engines.

    Returns
    -------
    FittedMVModel -- ``K`` is the latent diffusion matrix; ``correlation()`` gives
    the evolutionary correlations between genes.
    """
    is_ou = model == "OU"
    engine = resolve_estep(estep)
    init = obs.mode_init()
    p = init.shape[1]
    if trait_names is None:
        trait_names = list(range(p))
    n_reg = 1 if regimes is None else len(set(regimes.values()))
    root_regime = 0 if regimes is None else regimes[tree.root]
    n_L = p * (p + 1) // 2

    theta = np.tile(init.mean(axis=0), (n_reg, 1))
    K = np.atleast_2d(np.cov(init, rowvar=False) if p > 1 else [[max(np.var(init), 1e-2)]])
    K = K + 1e-3 * np.eye(p)
    alpha = float(alpha0) if is_ou else 0.0

    def theta_arg(th):
        return th if regimes is not None else th[0]

    prev_ll = -np.inf
    for it in range(max_em):
        # ---- E-step: posterior moments at current parameters (pluggable engine) ----
        es = engine.run(tree, obs, alpha if is_ou else 0.0, theta_arg(theta), K,
                        regimes=regimes, root_value=theta[root_regime])
        M, Z, Sig, cross = es["M"], es["Z"], es["Sigma"], es["cross"]

        m_pa = np.zeros_like(Z)
        Sig_pa = np.zeros((M.N, p, p))
        for i in range(M.N):
            pp = M.parent[i]
            if pp >= 0:
                m_pa[i] = Z[pp]
                Sig_pa[i] = Sig[pp]

        moments = {
            "m": jnp.asarray(Z), "Sig": jnp.asarray(Sig),
            "m_pa": jnp.asarray(m_pa), "Sig_pa": jnp.asarray(Sig_pa),
            "cross": jnp.asarray(cross), "t": jnp.asarray(M.t),
            "regime": jnp.asarray(M.regime_idx), "root_mask": jnp.asarray(M.is_root),
        }

        # ---- M-step: maximize Q via JAX-gradient BFGS ----
        neg_Q, split = _make_mstep_objective(moments, p, n_reg, is_ou)
        head = [np.log(max(alpha, 1e-4))] if is_ou else []
        x0 = jnp.asarray(np.concatenate([np.array(head), theta.reshape(-1), _pack_L(K, p)]))
        res = jax_minimize(neg_Q, x0, method="BFGS", options={"maxiter": mstep_maxiter})
        alpha_n, theta_n, L = split(res.x)
        alpha = float(alpha_n) if is_ou else 0.0
        theta = np.asarray(theta_n, dtype=float)
        K = np.asarray(L @ L.T, dtype=float)

        ll = mv_tree_laplace_marginal(tree, obs, alpha if is_ou else 0.0, theta_arg(theta), K,
                                      regimes=regimes, root_value=theta[root_regime])
        if verbose:
            print(f"  EM iter {it:2d}: marginal logL = {ll:.4f}")
        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol:
            prev_ll = ll
            break
        prev_ll = ll

    n = len(tree.root.get_leaves())
    n_params = (1 if is_ou else 0) + n_reg * p + n_L
    theta_out = (theta if regimes is not None else theta[0]) if is_ou else None
    return FittedMVModel("OU" if is_ou else "BM", list(trait_names), loglik=float(prev_ll),
                         n_params=n_params, n_obs=n, mu=theta[0].copy(), K=K,
                         alpha=(alpha if is_ou else None), theta=theta_out,
                         extra={"n_regimes": n_reg, "em_iters": it + 1})
