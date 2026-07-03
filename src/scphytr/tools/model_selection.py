"""Per-trait BM vs OU model selection on a phylogeny.

For each (univariate) trait we fit a Brownian-motion (BM) and a single-optimum
Ornstein-Uhlenbeck (OU-1) process by maximizing the linear-time pruning
likelihood, then compare them with AIC/BIC. An OU win is evidence of *adaptive*
evolution (pull toward an optimum) rather than neutral drift.

Fitting is univariate per trait, which is the standard setup for gene-/trait-wise
adaptive scans and keeps each likelihood evaluation O(n) over the tree.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..utils.pruning import (bm_pruning_logpdf, ou_pruning_logpdf,
                             bm_rates_pruning_logpdf, paint_regimes)
from ..inference.tree_laplace import latent_tree_laplace_marginal


@dataclass
class FittedModel:
    """Result of fitting one trait model: log-likelihood, parameter count, fit."""
    name: str
    loglik: float
    n_params: int
    n_obs: int
    params: dict = field(default_factory=dict)

    def aic(self):
        return 2.0 * self.n_params - 2.0 * self.loglik

    def aicc(self):
        # Small-sample-corrected AIC; falls back to AIC when n is large.
        k, n = self.n_params, self.n_obs
        denom = n - k - 1
        if denom <= 0:
            return np.inf
        return self.aic() + (2.0 * k * (k + 1)) / denom

    def bic(self):
        return self.n_params * np.log(self.n_obs) - 2.0 * self.loglik


def _set_single_trait(tree, values):
    """Attach a single trait named 'x' to leaves from a name->value mapping."""
    tree.set_trait_values({name: {"x": float(v)} for name, v in values.items()})


def fit_bm(tree, values, restarts=2, seed=0):
    """Fit Brownian motion to one trait (parameters: root mean mu, rate sigma2)."""
    _set_single_trait(tree, values)
    n = len(values)
    x = np.asarray(list(values.values()), dtype=float)

    def nll(p):
        mu, log_s2 = p
        return -bm_pruning_logpdf(tree, np.array([mu]), np.array([[np.exp(log_s2)]]))

    rng = np.random.default_rng(seed)
    var0 = max(np.var(x), 1e-6)
    inits = [np.array([np.mean(x), np.log(var0)])]
    inits += [np.array([np.mean(x) + rng.standard_normal(), np.log(var0) + rng.standard_normal()])
              for _ in range(restarts)]

    best = None
    for p0 in inits:
        res = minimize(nll, p0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res

    mu, log_s2 = best.x
    return FittedModel("BM", loglik=-best.fun, n_params=2, n_obs=n,
                       params={"mu": float(mu), "sigma2": float(np.exp(log_s2))})


def _tree_height(tree):
    """Maximum root-to-tip time (including the root branch)."""
    return float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)


def fit_ou(tree, values, alpha_inits=(0.1, 1.0, 5.0), seed=0):
    """Fit single-optimum OU to one trait (parameters: alpha, theta, sigma2).

    The fixed ancestral root is tied to the optimum (root_value = theta), which
    makes the marginal mean constant and resolves the root/optimum identifiability
    that plagues single-regime OU.

    ``alpha`` is capped so that ``alpha * tree_height <= 30``: beyond that the tips
    are effectively i.i.d. N(theta, sigma2/(2 alpha)) and the likelihood is flat in
    alpha (only the stationary variance is identified), so the cap avoids numerical
    underflow of the contraction without affecting model comparison.
    """
    _set_single_trait(tree, values)
    n = len(values)
    x = np.asarray(list(values.values()), dtype=float)
    alpha_max = 30.0 / max(_tree_height(tree), 1e-12)

    def nll(p):
        log_alpha, theta, log_s2 = p
        alpha = float(np.clip(np.exp(log_alpha), 1e-4, alpha_max))
        sigma2 = np.exp(log_s2)
        ll = ou_pruning_logpdf(tree, alpha, np.array([theta]),
                               np.array([[sigma2]]), root_value=np.array([theta]))
        return -ll if np.isfinite(ll) else 1e18

    var0 = max(np.var(x), 1e-6)
    mean0 = np.mean(x)
    best = None
    for alpha0 in alpha_inits:
        a0 = min(alpha0, alpha_max)
        # stationary var = sigma2 / (2 alpha)  =>  sigma2 ~ 2 alpha var0
        p0 = np.array([np.log(a0), mean0, np.log(2.0 * a0 * var0)])
        res = minimize(nll, p0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res

    log_alpha, theta, log_s2 = best.x
    alpha = float(np.clip(np.exp(log_alpha), 1e-4, alpha_max))
    return FittedModel("OU", loglik=-best.fun, n_params=3, n_obs=n,
                       params={"alpha": alpha, "theta": float(theta),
                               "sigma2": float(np.exp(log_s2))})


def fit_ou_regimes(tree, values, regimes, n_regimes, alpha_inits=(0.1, 1.0, 5.0), seed=0):
    """Fit multi-regime OU to one trait: shared alpha, per-regime optimum, sigma2.

    ``regimes`` maps each tree node to a regime id in [0, n_regimes); see
    ``scphytr.utils.pruning.paint_regimes``. The ancestral root is tied to the
    optimum of the root's regime. Parameter count is ``n_regimes + 2``, so AIC/BIC
    penalize each additional optimum.
    """
    _set_single_trait(tree, values)
    n = len(values)
    x = np.asarray(list(values.values()), dtype=float)
    alpha_max = 30.0 / max(_tree_height(tree), 1e-12)
    root_regime = regimes[tree.root]

    def nll(p):
        alpha = float(np.clip(np.exp(p[0]), 1e-4, alpha_max))
        thetas = np.asarray(p[1:1 + n_regimes]).reshape(n_regimes, 1)
        sigma2 = np.exp(p[-1])
        root_val = thetas[root_regime]
        ll = ou_pruning_logpdf(tree, alpha, thetas, np.array([[sigma2]]),
                               regimes=regimes, root_value=root_val)
        return -ll if np.isfinite(ll) else 1e18

    var0 = max(np.var(x), 1e-6)
    mean0 = np.mean(x)
    best = None
    for alpha0 in alpha_inits:
        a0 = min(alpha0, alpha_max)
        p0 = np.concatenate([[np.log(a0)], np.full(n_regimes, mean0), [np.log(2.0 * a0 * var0)]])
        res = minimize(nll, p0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res

    alpha = float(np.clip(np.exp(best.x[0]), 1e-4, alpha_max))
    thetas = np.asarray(best.x[1:1 + n_regimes], dtype=float)
    return FittedModel(f"OU{n_regimes}", loglik=-best.fun, n_params=n_regimes + 2, n_obs=n,
                       params={"alpha": alpha, "thetas": thetas.tolist(),
                               "sigma2": float(np.exp(best.x[-1])), "n_regimes": n_regimes})


# ---------------------------------------------------------------------------
# Clade-specific RATE heterogeneity: multi-rate Brownian motion (O'Meara's BMS).
# Unlike fit_ou_regimes (which varies the OU *optimum* per regime at a shared
# rate), these vary the diffusion *rate* sigma^2 per regime -- the model RevBayes
# fits with reversible-jump MCMC. fit_bm_rates scores a *given* regime painting;
# detect_rate_shifts searches for the shifts *de novo* by penalized likelihood.
# ---------------------------------------------------------------------------

def fit_bm_rates(tree, values, regimes, n_regimes, restarts=2, seed=0):
    """Fit multi-rate BM: one diffusion rate sigma^2 per painted regime.

    ``regimes`` maps each node to a regime id in [0, n_regimes) (the edge above the
    node; see :func:`paint_regimes`). The root mean is profiled out, so only the
    rates are optimized; the parameter count is ``n_regimes + 1`` (rates + root).
    """
    _set_single_trait(tree, values)
    n = len(values)
    x = np.asarray(list(values.values()), dtype=float)
    var0 = max(np.var(x), 1e-6)

    def nll(logr):
        ll = bm_rates_pruning_logpdf(tree, np.exp(logr), regimes)
        return -ll if np.isfinite(ll) else 1e18

    rng = np.random.default_rng(seed)
    inits = [np.log(np.full(n_regimes, var0))]
    inits += [np.log(np.full(n_regimes, var0)) + 0.5 * rng.standard_normal(n_regimes)
              for _ in range(restarts)]
    best = None
    for p0 in inits:
        res = minimize(nll, p0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    rates = np.exp(best.x)
    return FittedModel(f"BMS{n_regimes}", loglik=-best.fun, n_params=n_regimes + 1, n_obs=n,
                       params={"rates": rates.tolist(), "n_regimes": n_regimes})


def detect_rate_shifts(tree, values, max_shifts=5, criterion="bic", min_clade=5,
                       seed=0):
    """De-novo BM rate-shift detection by greedy penalized-likelihood selection.

    Starting from a single global rate, repeatedly add the branch whose rate shift
    most improves the (selection-corrected) ``criterion``, stopping when no shift
    helps. This is the maximum-penalized-likelihood counterpart to RevBayes' Bayesian
    reversible-jump search over the number and location of rate shifts: it returns a
    single best configuration (fast, O(n) per likelihood) rather than a posterior.

    Because each shift is the *best of* ~``m`` candidate branches, plain BIC badly
    over-selects (the maximum BIC drop over many tests is upward-biased). We add a
    location penalty of ``2 log m`` per shift -- a phylogenetic-BIC-style correction
    (cf. l1ou) for choosing where the shift goes -- which controls the null
    false-positive rate while retaining power on real shifts.

    Returns a dict with ``shifts`` (list of shift nodes, in the order added),
    ``regimes``/``n_regimes`` (final painting), ``fit`` (the chosen FittedModel),
    ``score_path`` (corrected criterion after each accepted shift) and ``baseline``.
    """
    _set_single_trait(tree, values)
    n = len(values)
    root = tree.root
    base_crit = (lambda m: m.bic()) if criterion == "bic" else (lambda m: m.aicc())

    candidates = [nd for nd in root.traverse()
                  if nd is not root and not nd.is_leaf()
                  and min_clade <= len(nd.get_leaves()) <= n - min_clade]
    loc_pen = 2.0 * np.log(max(len(candidates), 2))   # per-shift location penalty

    def crit(m, n_sh):
        return base_crit(m) + loc_pen * n_sh

    regimes0, nreg0 = paint_regimes(tree, [])
    base = fit_bm_rates(tree, values, regimes0, nreg0, seed=seed)
    best_fit, best_regimes, best_nreg = base, regimes0, nreg0
    shifts, score_path = [], [crit(base, 0)]
    cur = crit(base, 0)

    while len(shifts) < max_shifts:
        pick = None
        for nd in candidates:
            if nd in shifts:
                continue
            regimes, nreg = paint_regimes(tree, shifts + [nd])
            fm = fit_bm_rates(tree, values, regimes, nreg, restarts=1, seed=seed)
            sc = crit(fm, len(shifts) + 1)
            if pick is None or sc < pick[1]:
                pick = (nd, sc, fm, regimes, nreg)
        if pick is None or pick[1] >= cur - 1e-6:     # no shift improves the criterion
            break
        nd, cur, fm, best_regimes, best_nreg = pick
        shifts.append(nd)
        best_fit = fm
        score_path.append(cur)

    return {"shifts": shifts, "regimes": best_regimes, "n_regimes": best_nreg,
            "fit": best_fit, "baseline": base, "score_path": score_path,
            "criterion": criterion}


_FITTERS = {"BM": fit_bm, "OU": fit_ou}


# ---------------------------------------------------------------------------
# Count-observation variants: the leaf trait is a latent log-rate observed
# through a non-conjugate likelihood (e.g. Poisson counts). The marginal
# likelihood integrates out the latent value at *every* tree node via the O(n)
# tree-Laplace (see scphytr.inference.tree_laplace) -- never forming a dense
# covariance. The observation model only has to expose loglik / grad /
# neg_hess_diag, so any per-leaf likelihood plugs in. Each fit maximizes the
# marginal over the same hyperparameters as the Gaussian case, so AIC/BIC use
# the same parameter counts.
# ---------------------------------------------------------------------------

def fit_bm_counts(tree, obs, restarts=1, seed=0):
    """Fit BM to a latent trait observed through a non-conjugate likelihood."""
    from ..inference.tree_laplace import _TreeModel
    n = len(obs.mode_init())
    ydata = obs.mode_init()
    M = _TreeModel(tree, 0.0, 0.0, 1.0, root_value=0.0)   # structure cached; reused every eval

    def nll(p):
        mu, log_s2 = p
        try:
            ll = latent_tree_laplace_marginal(tree, obs, 0.0, mu, np.exp(log_s2), root_value=mu, model=M)
        except (np.linalg.LinAlgError, ValueError):
            return 1e18                       # degenerate (e.g. sigma2 -> 0); prune
        return -ll if np.isfinite(ll) else 1e18

    rng = np.random.default_rng(seed)
    var0 = max(np.var(ydata), 1e-3)
    inits = [np.array([np.mean(ydata), np.log(var0)])]
    inits += [np.array([np.mean(ydata) + rng.standard_normal(), np.log(var0) + rng.standard_normal()])
              for _ in range(restarts)]
    best = None
    for p0 in inits:
        res = minimize(nll, p0, method="L-BFGS-B",
                       options={"eps": 1e-4, "ftol": 1e-8, "gtol": 1e-5, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    mu, log_s2 = best.x
    return FittedModel("BM", loglik=-best.fun, n_params=2, n_obs=n,
                       params={"mu": float(mu), "sigma2": float(np.exp(log_s2))})


def fit_ou_counts(tree, obs, alpha_inits=(0.1, 1.0, 5.0), seed=0):
    """Fit single-optimum OU to a latent trait observed non-conjugately."""
    from ..inference.tree_laplace import _TreeModel
    n = len(obs.mode_init())
    alpha_max = _ou_alpha_max(tree)
    ydata = obs.mode_init()
    M = _TreeModel(tree, 0.1, 0.0, 1.0, root_value=0.0)   # structure cached; reused every eval

    def nll(p):
        alpha = float(np.clip(np.exp(p[0]), 1e-4, alpha_max))
        theta, log_s2 = p[1], p[2]
        try:
            ll = latent_tree_laplace_marginal(tree, obs, alpha, theta, np.exp(log_s2),
                                              root_value=theta, model=M)
        except (np.linalg.LinAlgError, ValueError):
            return 1e18
        return -ll if np.isfinite(ll) else 1e18

    var0 = max(np.var(ydata), 1e-3)
    mean0 = np.mean(ydata)
    best = None
    for alpha0 in alpha_inits:
        a0 = min(alpha0, alpha_max)
        p0 = np.array([np.log(a0), mean0, np.log(2.0 * a0 * var0)])
        res = minimize(nll, p0, method="L-BFGS-B",
                       options={"eps": 1e-4, "ftol": 1e-8, "gtol": 1e-5, "maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    alpha = float(np.clip(np.exp(best.x[0]), 1e-4, alpha_max))
    return FittedModel("OU", loglik=-best.fun, n_params=3, n_obs=n,
                       params={"alpha": alpha, "theta": float(best.x[1]),
                               "sigma2": float(np.exp(best.x[2]))})


def fit_ou_regimes_counts(tree, obs, regimes, n_regimes,
                          alpha_inits=(0.1, 1.0, 5.0), seed=0):
    """Fit multi-regime OU to a latent trait observed non-conjugately."""
    n = len(obs.mode_init())
    alpha_max = _ou_alpha_max(tree)
    root_regime = regimes[tree.root]
    ydata = obs.mode_init()

    def nll(p):
        alpha = float(np.clip(np.exp(p[0]), 1e-4, alpha_max))
        thetas = np.asarray(p[1:1 + n_regimes], dtype=float)
        log_s2 = p[-1]
        try:
            ll = latent_tree_laplace_marginal(tree, obs, alpha, thetas, np.exp(log_s2),
                                              regimes=regimes, root_value=thetas[root_regime])
        except (np.linalg.LinAlgError, ValueError):
            return 1e18
        return -ll if np.isfinite(ll) else 1e18

    var0 = max(np.var(ydata), 1e-3)
    mean0 = np.mean(ydata)
    best = None
    for alpha0 in alpha_inits:
        a0 = min(alpha0, alpha_max)
        p0 = np.concatenate([[np.log(a0)], np.full(n_regimes, mean0), [np.log(2.0 * a0 * var0)]])
        res = minimize(nll, p0, method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    alpha = float(np.clip(np.exp(best.x[0]), 1e-4, alpha_max))
    thetas = np.asarray(best.x[1:1 + n_regimes], dtype=float)
    return FittedModel(f"OU{n_regimes}", loglik=-best.fun, n_params=n_regimes + 2, n_obs=n,
                       params={"alpha": alpha, "thetas": thetas.tolist(),
                               "sigma2": float(np.exp(best.x[-1])), "n_regimes": n_regimes})


def detect_adaptive_counts(tree, counts_table, size_factors, models=("BM", "OU"),
                           criterion="aic", regimes=None):
    """BM vs OU adaptive detection for genes observed as Poisson counts.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    counts_table : pandas.DataFrame
        Rows indexed by leaf/species name, one column per gene; entries are total
        counts aggregated over the cells of each species/clone.
    size_factors : pandas.Series or dict
        Per-species summed size factors (offsets), indexed by leaf name.
    models : tuple[str]
        Subset of {"BM", "OU", "OU2"}; "OU2" requires ``regimes``.
    criterion : {"aic", "aicc", "bic"}
    regimes : dict[node -> int], optional

    Returns
    -------
    pandas.DataFrame indexed by gene (same columns as ``detect_adaptive``).
    """
    from ..inference.laplace import PoissonObservation

    leaf_names = tree.phylotree.get_leaf_names()
    missing = set(leaf_names) - set(counts_table.index)
    if missing:
        raise ValueError(f"counts_table is missing leaves: {sorted(missing)}")
    S = np.array([float(size_factors[name]) for name in leaf_names])

    if "OU2" in models:
        if regimes is None:
            raise ValueError("model 'OU2' requires a `regimes` painting (see paint_regimes).")
        n_regimes = len(set(regimes.values()))
        ou2_name = f"OU{n_regimes}"

    def fit_one(model, obs):
        if model == "BM":
            return fit_bm_counts(tree, obs)
        if model == "OU":
            return fit_ou_counts(tree, obs)
        if model == "OU2":
            return fit_ou_regimes_counts(tree, obs, regimes, n_regimes)
        raise ValueError(f"Unknown model '{model}'")

    rows = []
    for gene in counts_table.columns:
        Y = np.array([float(counts_table.loc[name, gene]) for name in leaf_names])
        obs = PoissonObservation(Y, S)
        fitted = [fit_one(m, obs) for m in models]
        row, _ = _selection_row(fitted, criterion)
        by_name = {m.name: m for m in fitted}
        if "OU" in by_name:
            row["alpha"] = by_name["OU"].params["alpha"]
            row["theta"] = by_name["OU"].params["theta"]
            row["sigma2_OU"] = by_name["OU"].params["sigma2"]
        if "BM" in by_name:
            row["sigma2_BM"] = by_name["BM"].params["sigma2"]
        if "OU2" in models:
            row["alpha_OU2"] = by_name[ou2_name].params["alpha"]
            row["thetas_OU2"] = by_name[ou2_name].params["thetas"]
        rows.append(row)

    return pd.DataFrame(rows, index=pd.Index(counts_table.columns, name="gene"))


def _ou_alpha_max(tree):
    return 30.0 / max(_tree_height(tree), 1e-12)


def _selection_row(fitted, criterion):
    """Common per-trait result row: selection, criterion gap, per-model scores."""
    best, scores = select_model(fitted, criterion=criterion)
    ordered = sorted(scores.values())
    gap = (ordered[1] - ordered[0]) if len(ordered) > 1 else np.nan
    row = {"selected": best.name, f"d_{criterion}": gap, "adaptive": best.name.startswith("OU")}
    for m in fitted:
        row[f"loglik_{m.name}"] = m.loglik
        row[f"{criterion}_{m.name}"] = scores[m.name]
    return row, best


def select_model(models, criterion="aic"):
    """Pick the model with the lowest information criterion.

    Parameters
    ----------
    models : list[FittedModel]
    criterion : {"aic", "aicc", "bic"}

    Returns
    -------
    (best_model, scores) where scores maps model name -> criterion value.
    """
    score = {"aic": lambda m: m.aic(), "aicc": lambda m: m.aicc(), "bic": lambda m: m.bic()}[criterion]
    scores = {m.name: score(m) for m in models}
    best = min(models, key=score)
    return best, scores


def detect_adaptive(tree, trait_table, models=("BM", "OU"), criterion="aic", regimes=None):
    """Fit and compare models for every trait; flag adaptive (OU-selected) traits.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    trait_table : pandas.DataFrame
        Rows indexed by leaf/species name, one column per trait.
    models : tuple[str]
        Subset of {"BM", "OU", "OU2"} to fit and compare. "OU2" is multi-regime
        OU and requires ``regimes``.
    criterion : {"aic", "aicc", "bic"}
    regimes : dict[node -> int], optional
        Per-node regime painting for the "OU2" model (see ``paint_regimes``).

    Returns
    -------
    pandas.DataFrame indexed by trait, with per-model log-likelihood and
    criterion, the selected model, the criterion gap to the runner-up, fitted
    parameters, and a boolean ``adaptive`` flag (any OU model selected).
    """
    leaf_names = tree.phylotree.get_leaf_names()
    missing = set(leaf_names) - set(trait_table.index)
    if missing:
        raise ValueError(f"trait_table is missing values for leaves: {sorted(missing)}")

    if "OU2" in models:
        if regimes is None:
            raise ValueError("model 'OU2' requires a `regimes` painting (see paint_regimes).")
        n_regimes = len(set(regimes.values()))
        ou2_name = f"OU{n_regimes}"

    def fit_one(model, values):
        if model == "OU2":
            return fit_ou_regimes(tree, values, regimes, n_regimes)
        return _FITTERS[model](tree, values)

    rows = []
    for trait in trait_table.columns:
        values = {name: trait_table.loc[name, trait] for name in leaf_names}
        fitted = [fit_one(m, values) for m in models]
        row, _ = _selection_row(fitted, criterion)

        by_name = {m.name: m for m in fitted}
        if "OU" in by_name:
            row["alpha"] = by_name["OU"].params["alpha"]
            row["theta"] = by_name["OU"].params["theta"]
            row["sigma2_OU"] = by_name["OU"].params["sigma2"]
        if "BM" in by_name:
            row["sigma2_BM"] = by_name["BM"].params["sigma2"]
        if "OU2" in models:
            row["alpha_OU2"] = by_name[ou2_name].params["alpha"]
            row["thetas_OU2"] = by_name[ou2_name].params["thetas"]
        rows.append(row)

    return pd.DataFrame(rows, index=pd.Index(trait_table.columns, name="trait"))
