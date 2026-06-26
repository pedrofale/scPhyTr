"""Laplace inference as a first-class, modular algorithm.

``Laplace`` composes any *trait model* (the latent BM/OU/multi-rate process) with any
*observation model* (Poisson / negative-binomial / Gaussian, with cells kept as subclonal
replicates) and computes the marginal likelihood by the existing linear-time pruning /
sparsity-preserving Laplace engines (:mod:`scphytr.inference.tree_laplace`,
:mod:`scphytr.utils.pruning`). It is the fast default backend behind the ``scphytr.tl`` API,
and reproduces the standalone ``tools.model_selection`` / ``tools.heritability`` fits.

Contract
--------
* trait model exposes ``process_params()`` -> dict(alpha, theta, sigma2, regimes, root_value,
  rates) and ``learnable`` / ``pack()`` / ``unpack(x)`` for the scalar hyper-parameters;
* observation model is either ``None`` (directly observed Gaussian trait -> exact pruning) or a
  leaf-likelihood object exposing ``mode_init/loglik/grad/neg_hess_diag`` (the subclonal count
  engine, :class:`scphytr.inference.laplace.MultiCellPoissonObservation`).
"""
import numpy as np
from scipy.optimize import minimize

from .base import BaseInference
from .tree_laplace import latent_tree_laplace_marginal
from ..utils.pruning import bm_pruning_logpdf, ou_pruning_logpdf, bm_rates_pruning_logpdf


class Laplace(BaseInference):
    """Sparsity-preserving Laplace / exact-pruning inference over (trait, observation)."""

    def __init__(self, tree, trait_model, observation_model=None):
        # Tolerate either a BaseObservationModel or a raw leaf-likelihood object
        # (the subclonal count engine: mode_init/loglik/grad/neg_hess_diag).
        self.tree = tree
        self.trait_model = trait_model
        self.observation_model = observation_model
        self.trait_learnable_parameters = getattr(trait_model, "learnable_parameters", None)
        self.observation_learnable_parameters = (
            observation_model.get_learnable_parameters()
            if hasattr(observation_model, "get_learnable_parameters") else None)

    # ----- marginal likelihood for the current trait-model parameters -----------------
    def marginal_loglik(self):
        p = self.trait_model.process_params()
        obs = self.observation_model
        if obs is None:
            # directly observed (Gaussian) trait: exact Felsenstein pruning
            if p.get("rates") is not None:                       # multi-rate BM
                return bm_rates_pruning_logpdf(self.tree, p["rates"], p["regimes"],
                                               mu=p.get("root_value"))
            if p["alpha"] is None or p["alpha"] <= 0:            # BM
                mu = np.atleast_1d(p["theta"])
                return bm_pruning_logpdf(self.tree, mu, np.atleast_2d(p["sigma2"]))
            return ou_pruning_logpdf(self.tree, p["alpha"], np.atleast_1d(p["theta"]),
                                     np.atleast_2d(p["sigma2"]), regimes=p["regimes"],
                                     root_value=p.get("root_value"))
        # non-conjugate (count) observation: O(n) latent tree-Laplace
        return latent_tree_laplace_marginal(
            self.tree, obs, alpha=p["alpha"], theta=p["theta"], sigma2=p["sigma2"],
            regimes=p["regimes"], root_value=p.get("root_value"))

    # ----- maximum-marginal-likelihood fit of the learnable hyper-parameters ----------
    def fit_trait_model(self, restarts=2, seed=0):
        x0 = self.trait_model.pack()
        if x0.size == 0:
            return self.trait_model

        def nll(x):
            self.trait_model.unpack(x)
            ll = self.marginal_loglik()
            return -ll if np.isfinite(ll) else 1e18

        rng = np.random.default_rng(seed)
        best = None
        for r in range(restarts + 1):
            xi = x0 if r == 0 else x0 + 0.3 * rng.standard_normal(x0.shape)
            res = minimize(nll, xi, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
            if best is None or res.fun < best.fun:
                best = res
        self.trait_model.unpack(best.x)
        self.loglik_ = -best.fun
        return self.trait_model

    # convenience
    def fit(self, **kw):
        self.fit_trait_model(**kw)
        return {"trait_model": self.trait_model, "loglik": getattr(self, "loglik_", None)}
