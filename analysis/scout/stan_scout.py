"""Driver for ``stan/scout.stan`` -- SCOUT's per-gene model comparison, done with a posterior.

SCOUT reports one word per gene ("BM1" / "OU1" / "OUx") from a min-AICc argmax, with no statement
of how sure it is; at 32 leaves that word is right 55% of the time. This fits the same three
hypotheses on the same supplied regimes and returns ``P(model | y)`` plus posteriors for alpha,
sigma and the optima.

The tree never enters Stan. Each root-to-tip path is precomputed here as a list of
``(tip, regime, start, end)`` segments, from which Stan rebuilds the OU regime-weight matrix

    W[i, k](alpha) = sum over tip i's segments in regime k of
                     exp(-alpha (T_i - end)) - exp(-alpha (T_i - start))

plus ``exp(-alpha T_i)`` on the root regime. :func:`regime_weights` is the numpy twin of that loop
and is checked against :func:`scphytr.modes.baseline.regime_design`.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scphytr.modes._ou import tip_cov
from analysis.scout.scout_preprocess import patristic
from scphytr.modes.baseline import paint_regimes

__all__ = ["regime_segments", "regime_weights", "stan_data", "get_model", "default_inits",
           "fit_gene", "fit_genes"]

MODELS = ("BM1", "OU1", "OUX")
STAN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "stan", "scout.stan")


def regime_segments(tree, node_regime):
    """Merged ``(tip, regime, start, end)`` segments along every root-to-tip path (0-based)."""
    depth = tree.depth
    segs = []
    for i, leaf in enumerate(tree.leaves):
        path = []
        v = int(leaf)
        while tree.parent[v] >= 0:
            path.append(v)
            v = int(tree.parent[v])
        path.reverse()
        for v in path:
            k = int(node_regime[v])
            s, e = float(depth[tree.parent[v]]), float(depth[v])
            if segs and segs[-1][0] == i and segs[-1][1] == k and np.isclose(segs[-1][3], s):
                segs[-1][3] = e                       # merge consecutive same-regime branches
            else:
                segs.append([i, k, s, e])
    return np.array(segs, dtype=float)


def regime_weights(alpha, segs, tip_depth, n, K, root_reg):
    """numpy twin of the Stan loop; ``tip_mean = W @ theta``."""
    W = np.zeros((n, K))
    for i, k, s, e in segs:
        i, k = int(i), int(k)
        W[i, k] += np.exp(-alpha * (tip_depth[i] - e)) - np.exp(-alpha * (tip_depth[i] - s))
    W[:, root_reg] += np.exp(-alpha * tip_depth)
    return W


def stan_data(tree, y, leaf_regime, use_tau=True, s_theta=1.0, s_scale=1.0, s_tau=0.5,
              sd_log_hl=1.0, model_prior=(1 / 3, 1 / 3, 1 / 3), fit_model=0, standardize=True):
    """Assemble the Stan data block for one gene."""
    y = np.asarray(y, dtype=float).ravel()
    if standardize:
        y = (y - y.mean()) / (y.std() + 1e-12)
    node_regime, uniq = paint_regimes(tree, leaf_regime)
    K = len(uniq)
    segs = regime_segments(tree, node_regime)
    tip_depth = tree.depth[tree.leaves]
    C = tip_cov(tree, 1e-12, 1.0, root="fixed")
    return dict(
        n=tree.n_leaves, K=K, y=y.tolist(), D=patristic(tree).tolist(), C=C.tolist(),
        tip_depth=tip_depth.tolist(), n_seg=len(segs),
        seg_tip=(segs[:, 0].astype(int) + 1).tolist(), seg_reg=(segs[:, 1].astype(int) + 1).tolist(),
        seg_s=segs[:, 2].tolist(), seg_e=segs[:, 3].tolist(),
        root_reg=int(node_regime[0]) + 1, tree_height=float(tip_depth.max()),
        use_tau=int(use_tau), s_theta=float(s_theta), s_scale=float(s_scale), s_tau=float(s_tau),
        sd_log_hl=float(sd_log_hl), log_model_prior=np.log(model_prior).tolist(),
        fit_model=int(fit_model)), uniq


def get_model(stan_file=STAN_FILE):
    from cmdstanpy import CmdStanModel
    return CmdStanModel(stan_file=stan_file)


def _summarise(fit, uniq):
    dr = fit.stan_variables()
    p = dr["p_model"].mean(axis=0)
    return dict(
        p_BM1=float(p[0]), p_OU1=float(p[1]), p_OUX=float(p[2]),
        call=MODELS[int(np.argmax(p))], p_max=float(p.max()),
        alpha_ou1=float(np.median(dr["alpha_ou1"])), alpha_oux=float(np.median(dr["alpha_oux"])),
        sigma_bm=float(np.median(dr["sigma_bm"])), sigma_oux=float(np.median(dr["sigma_oux"])),
        theta_spread=float(np.median(dr["theta_spread"])),
        theta_oux=np.median(dr["theta_oux"], axis=0).tolist(), regimes=list(uniq),
        max_rhat=float(np.nanmax(fit.summary()["R_hat"].values)),
        min_ess=float(np.nanmin(fit.summary()["ESS_bulk"].values)),
        divergences=int(fit.method_variables()["divergent__"].sum()))


def default_inits(K, use_tau=True, chains=4, seed=0):
    """One jittered init per chain, around unit scales / half-life ~ one tree height / no shift.

    Two failure modes this avoids. Stan's own random inits wander into alpha extremes during warmup
    and throw (recoverable) Cholesky errors. But giving every chain the SAME init is worse: R-hat
    then cannot see between-chain disagreement, and reports convergence that has not happened -- an
    identical-init run of the 256-tip fit reported R-hat 1.09 where jittered inits give 1.01.
    """
    out = []
    for c in range(chains):
        r = np.random.default_rng(seed * 1000 + c)
        d = dict(theta_bm=float(r.normal(0, 0.3)), sigma_bm=float(np.exp(r.normal(0, 0.3))),
                 theta_ou1=float(r.normal(0, 0.3)), s_ou1=float(np.exp(r.normal(0, 0.3))),
                 hl_ou1=float(np.exp(r.normal(0, 0.5))), theta_oux=r.normal(0, 0.3, K).tolist(),
                 s_oux=float(np.exp(r.normal(0, 0.3))), hl_oux=float(np.exp(r.normal(0, 0.5))))
        d["tau"] = [0.3, 0.3, 0.3] if use_tau else []
        out.append(d)
    return out


def fit_gene(tree, y, leaf_regime, model=None, chains=4, iter_sampling=1000, iter_warmup=1000,
             seed=1, show_progress=False, inits=None, adapt_delta=0.9, **kw):
    """Fit one gene. Returns ``(summary_dict, CmdStanMCMC)``.

    ``adapt_delta`` defaults above Stan's 0.8: the OU alpha-sigma ridge produces a low rate of
    divergences at 0.8 (16/2000 at 256 tips) which 0.99 removes entirely, at 2.4x the runtime.
    """
    model = get_model() if model is None else model
    data, uniq = stan_data(tree, y, leaf_regime, **kw)
    if inits is None:
        inits = default_inits(data["K"], bool(data["use_tau"]), chains=chains, seed=seed)
    fit = model.sample(data=data, chains=chains, iter_sampling=iter_sampling,
                       iter_warmup=iter_warmup, seed=seed, show_progress=show_progress,
                       inits=inits, adapt_delta=adapt_delta, show_console=False)
    return _summarise(fit, uniq), fit


def _one(args):
    tree, y, leaf_regime, kw, seed = args
    s, _ = fit_gene(tree, y, leaf_regime, seed=seed, **kw)
    return s


def fit_genes(Y, tree, leaf_regime, n_jobs=None, seed=1, **kw):
    """Fit every column of ``Y`` independently. Returns a DataFrame, one row per gene."""
    import pandas as pd
    Y = np.asarray(Y, dtype=float)
    get_model()                                        # compile once before forking
    jobs = [(tree, Y[:, g], leaf_regime, kw, seed + g) for g in range(Y.shape[1])]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        rows = list(ex.map(_one, jobs))
    return pd.DataFrame(rows)
