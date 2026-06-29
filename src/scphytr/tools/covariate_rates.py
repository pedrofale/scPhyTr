"""Covariate-associated rate shifts: does a discrete covariate carry its own evolutionary rate?

Given a discrete label on the leaves (e.g. ``clone`` or spatial ``niche``), we reconstruct the
covariate over the internal branches (Fitch parsimony) and fit a **state-dependent multi-rate
Brownian motion** -- one diffusion rate sigma^2 per covariate state -- then test it against a
single global rate by a likelihood-ratio test. This is the maximum-likelihood counterpart to
RevBayes' state-dependent BM (a discrete character mapped on the tree, the continuous rate
indexed by the mapped state): we paint the branches by the same parsimony reconstruction and
estimate the per-state rates directly, in linear time, with a calibrated LRT instead of MCMC.

Unlike :func:`scphytr.tools.model_selection.detect_rate_shifts` (which *searches* for clade
shifts de novo), here the regime partition is **given by the covariate**, so there is no
location penalty -- the test has a clean ``n_states - 1`` degrees of freedom.
"""
import numpy as np
from scipy.stats import chi2

from .model_selection import fit_bm, fit_bm_rates


def reconstruct_states(tree, leaf_labels):
    """Fitch parsimony reconstruction of a discrete leaf label over every node.

    ``leaf_labels`` maps leaf name -> state (any hashable). Returns
    ``(regimes, n_regimes, state_names)`` where ``regimes`` maps each node to a regime id in
    ``[0, n_regimes)`` (the state assigned to the branch above the node) and ``state_names[i]``
    is the original label of regime ``i``. The assignment is a most-parsimonious one (down/up
    pass); ties are broken deterministically by the global state order.
    """
    states = sorted({leaf_labels[l.name] for l in tree.root.get_leaves()}, key=str)
    idx = {s: i for i, s in enumerate(states)}

    # ---- up pass (post-order): Fitch sets -------------------------------------------------
    fset = {}
    for nd in reversed(list(tree.root.traverse("levelorder"))):
        if nd.is_leaf():
            fset[nd] = {leaf_labels[nd.name]}
        else:
            child_sets = [fset[c] for c in nd.children]
            inter = set.intersection(*child_sets)
            fset[nd] = inter if inter else set.union(*child_sets)

    # ---- down pass (pre-order): resolve to a single state ---------------------------------
    assign = {}
    root = tree.root
    assign[root] = min(fset[root], key=lambda s: idx[s])
    for nd in root.traverse("preorder"):
        if nd is root:
            continue
        parent_state = assign[nd.up]
        assign[nd] = parent_state if parent_state in fset[nd] \
            else min(fset[nd], key=lambda s: idx[s])

    regimes = {nd: idx[assign[nd]] for nd in root.traverse()}
    return regimes, len(states), states


def fit_covariate_rates(tree, values, leaf_labels, restarts=2, seed=0):
    """Test whether a discrete covariate carries state-specific BM rates.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    values : dict leaf name -> trait value (the trait whose rate is tested, e.g. a gene's
        per-leaf mean log-expression).
    leaf_labels : dict leaf name -> discrete state (the covariate, e.g. clone or niche).

    Returns a dict with the per-state ``rates`` (state label -> sigma^2), the single-rate
    ``null_rate``, the log-likelihoods, the LRT statistic / df / ``p`` value, AIC for both
    models, the ``rate_ratio`` (max/min state rate), the fastest state, and the branch
    ``regimes``/``state_names`` (for painting the tree).
    """
    regimes, n_regimes, state_names = reconstruct_states(tree, leaf_labels)
    null = fit_bm(tree, values, restarts=restarts, seed=seed)
    if n_regimes < 2:                                  # only one state present -> no test
        return {"rates": {state_names[0]: null.params["sigma2"]},
                "null_rate": null.params["sigma2"], "n_states": 1,
                "loglik_null": null.loglik, "loglik_full": null.loglik,
                "lrt_stat": 0.0, "lrt_df": 0, "p": 1.0,
                "aic_null": null.aic(), "aic_full": null.aic(),
                "rate_ratio": 1.0, "fastest_state": state_names[0],
                "regimes": regimes, "state_names": state_names}
    full = fit_bm_rates(tree, values, regimes, n_regimes, restarts=restarts, seed=seed)
    rates = np.asarray(full.params["rates"], dtype=float)
    stat = max(2.0 * (full.loglik - null.loglik), 0.0)
    df = n_regimes - 1
    p = float(chi2.sf(stat, df))
    rate_map = {state_names[i]: float(rates[i]) for i in range(n_regimes)}
    return {"rates": rate_map, "null_rate": float(null.params["sigma2"]), "n_states": n_regimes,
            "loglik_null": float(null.loglik), "loglik_full": float(full.loglik),
            "lrt_stat": float(stat), "lrt_df": int(df), "p": p,
            "aic_null": float(null.aic()), "aic_full": float(full.aic()),
            "rate_ratio": float(rates.max() / max(rates.min(), 1e-12)),
            "fastest_state": state_names[int(np.argmax(rates))],
            "regimes": regimes, "state_names": state_names}
