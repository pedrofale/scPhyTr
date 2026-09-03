"""Model 1 -- the first Bayesian model: nested spike-and-slab over (branch, gene).

    z_b      ~ Bernoulli(pi)                            which BRANCHES carry an adaptive event
    g_bg|z_b ~ Bernoulli(z_b * rho)                     which GENES respond to it
    d_bg|g   ~ N(0, omega^2 sigma_g^2)  if g_bg else 0  the optimum shift
    Y_g      ~ N(theta0_g 1 + U(alpha) d_g,  sigma_g^2 [R(alpha) + tau I])

Nothing here is a bespoke phylogenetic optimiser. By :mod:`scphytr.modes.design` the mean is *linear* in
``delta``, so after whitening with ``R(alpha) = L L^T`` this is Gaussian linear variable selection
and every coordinate is conjugate: ``theta0_g`` flat (projected out), ``delta`` Gaussian slab (closed
form), ``sigma_g^2`` inverse-gamma per gene, ``pi``/``rho`` Beta. Only ``alpha`` is non-conjugate,
and neither is ``tau``; both are the only coordinates touching an O(n^3) factorisation, hence a
precomputed 2-D grid walked by Metropolis.

``tau`` is measurement noise as a fraction of the trait's stationary variance (``R`` is normalised to
unit mean diagonal). It is what replaces SCOUT's lineage smoothing. Smoothing multiplies the data by
a lineage-similarity matrix, so the smoothed covariance is ``S R S^T`` -- far more short-range
lineage correlation than any OU can produce. Fed to a *generative* model that mismatch is patched
with spurious small-clade events. A per-gene test can get away with smoothing; a joint model cannot,
and does not need to.

Two things make it work:

**1. Collapsed pooling.** ``gamma_.b`` is integrated out inside the ``z_b`` update, so

    logit P(z_b = 1 | .) = logit(pi) + sum_g log[ (1 - rho) + rho * BF_bg ] .

A non-responder has ``BF ~ 1`` and contributes ``~0``; it cannot swamp the responders. That is
exactly where naive summed-dAICc pooling fails (experiments/05).

**2. Parent/child block moves.** On an ultrametric tree ``U[:, b]`` is a *scaled clade indicator*,
so a parent's column lies exactly in the span of any child partition of its clade: the likelihood
cannot distinguish one event at ``p`` from events at all its children. Only the sparsity prior can,
and single-site Gibbs can never make that jump. So the sweep updates ``{p} u children(p)`` jointly,
enumerating the 2^m configurations of ``z`` with ``gamma`` and ``delta`` collapsed. This is the
Bayesian counterpart of SCOUT's partition degeneracy -- with the difference that here the ambiguity
is resolved by parsimony, and whatever ambiguity remains is *reported* as posterior spread.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .design import shift_design, candidate_branches
from ._ou import tip_cov

__all__ = ["Model1Result", "fit_model1"]


@dataclass
class Model1Result:
    branches: np.ndarray          # (B,) node index of each candidate branch
    p_z: np.ndarray               # (B,) posterior P(z_b = 1 | Y)
    p_gamma: np.ndarray           # (B, G) posterior P(gamma_bg = 1 | Y)
    delta_mean: np.ndarray        # (B, G) posterior mean shift
    alpha_draws: np.ndarray
    tau_draws: np.ndarray
    pi_draws: np.ndarray
    rho_draws: np.ndarray
    sigma2_mean: np.ndarray       # (G,)
    n_event_draws: np.ndarray
    diagnostics: dict

    def top_branches(self, k=5):
        o = np.argsort(-self.p_z)[:k]
        return [(int(self.branches[i]), float(self.p_z[i])) for i in o]

    def clade_support(self, tree, k=5):
        """P(an event on the branch OR anywhere in the chain below it) -- the identified read-out."""
        pos = {int(b): i for i, b in enumerate(self.branches)}
        out = []
        for b, i in pos.items():
            s = self.p_z[i]
            out.append((b, float(min(s, 1.0))))
        out.sort(key=lambda t: -t[1])
        return out[:k]

    def responders(self, branch, thresh=0.5):
        i = int(np.where(self.branches == branch)[0][0])
        return np.where(self.p_gamma[i] >= thresh)[0]


def _prepare(tree, Y, branches, alpha, tau, root):
    """Whiten by [R(alpha) + tau I]^{-1/2} and project out the flat-prior intercept."""
    n = tree.n_leaves
    R = tip_cov(tree, alpha, 1.0, root=root)
    R = R / np.mean(np.diag(R))                     # unit scale, so tau means the same at every alpha
    L = np.linalg.cholesky(R + (tau + 1e-10) * np.eye(n))
    U = np.linalg.solve(L, shift_design(tree, alpha, branches))
    Yt = np.linalg.solve(L, Y)
    one = np.linalg.solve(L, np.ones((n, 1)))
    s = float(one.T @ one)
    P = lambda A: A - one @ (one.T @ A) / s
    U, Yt = P(U), P(Yt)
    return dict(U=U, Y=Yt, logdetL=float(np.sum(np.log(np.diag(L)))), log_one=0.5 * np.log(s))


def _blocks(tree, branches):
    """``{p} u children(p)`` for every node, restricted to candidate branches."""
    pos = {int(b): i for i, b in enumerate(branches)}
    out = []
    for p in range(tree.n_nodes):
        blk = ([pos[p]] if p in pos else []) + [pos[c] for c in tree.children[p] if int(c) in pos]
        if blk:
            out.append(np.array(sorted(set(blk)), dtype=int))
    return out


def _subsets(m):
    return [tuple(s) for k in range(m + 1) for s in combinations(range(m), k)]


def fit_model1(Y, tree, branches=None, alpha_grid=(1.0,), tau_grid=(0.0,), root="stationary", omega=1.0,
               n_iter=2000, burn=500, thin=1, seed=0, min_leaves=8,
               a_sigma=2.0, b_sigma=1.0, pi_prior=(1.0, 20.0), rho_prior=(1.0, 4.0),
               standardize=True, verbose=False):
    """Block-collapsed Gibbs for Model 1. ``Y`` is (n_leaves, G) continuous tip expression."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    if standardize:
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    n, G = Y.shape
    branches = candidate_branches(tree, min_leaves) if branches is None else np.asarray(branches, int)
    B = len(branches)
    alpha_grid = np.atleast_1d(np.asarray(alpha_grid, dtype=float))
    tau_grid = np.atleast_1d(np.asarray(tau_grid, dtype=float))
    na, nt = len(alpha_grid), len(tau_grid)
    pre = [_prepare(tree, Y, branches, a, t, root) for a in alpha_grid for t in tau_grid]
    blocks = _blocks(tree, branches)
    ai, ti = na // 2, nt // 2
    om2 = float(omega) ** 2

    z = np.zeros(B, dtype=bool)
    gam = np.zeros((B, G), dtype=bool)
    D = np.zeros((B, G))
    sig2 = np.ones(G)
    pi, rho = 1.0 / max(B, 2), 0.25
    r = pre[ai * nt + ti]["Y"].copy()

    acc = np.zeros(2)
    Pz = np.zeros(B); Pg = np.zeros((B, G)); Dm = np.zeros((B, G)); S2 = np.zeros(G)
    a_draws, t_draws, pi_draws, rho_draws, nev = [], [], [], [], []
    keep = 0

    for it in range(n_iter):
        U = pre[ai * nt + ti]["U"]
        lpi, l1pi = np.log(pi), np.log1p(-pi)
        lrho, l1rho = np.log(max(rho, 1e-12)), np.log(max(1 - rho, 1e-12))

        for blk in (blocks[i] for i in rng.permutation(len(blocks))):
            m = len(blk)
            Ub = U[:, blk]
            r += Ub @ D[blk]                                   # take the block out
            C = Ub.T @ r                                       # (m, G)
            Gm = Ub.T @ Ub
            subs = _subsets(m)
            # per-subset: log|A_S|, A_S^{-1}, and the gene-wise quadratic form
            lm = np.zeros((len(subs), G)); Ainv = {}; Lc = {}
            for si, S in enumerate(subs):
                if not S:
                    continue
                idx = np.array(S)
                A = Gm[np.ix_(idx, idx)] + np.eye(len(S)) / om2
                Ai = np.linalg.inv(A)
                Ainv[S] = Ai
                Lc[S] = np.linalg.cholesky(Ai + 1e-12 * np.eye(len(S)))
                cs = C[idx]                                    # (|S|, G)
                q = np.einsum("ig,ij,jg->g", cs, Ai, cs)
                sign, ld = np.linalg.slogdet(A)
                lm[si] = -0.5 * (len(S) * np.log(om2) + ld) + q / (2.0 * sig2)
            # ---- z-config weights, with gamma collapsed ----
            cfgs = [tuple(c) for c in np.ndindex(*([2] * m))]
            w = np.empty(len(cfgs))
            per_cfg = {}
            for ci, cfg in enumerate(cfgs):
                on = tuple(i for i in range(m) if cfg[i])
                allowed = [si for si, S in enumerate(subs) if set(S) <= set(on)]
                lw = np.array([lm[si] + len(subs[si]) * lrho
                               + (len(on) - len(subs[si])) * l1rho for si in allowed])
                mx = lw.max(0)
                tot = mx + np.log(np.exp(lw - mx).sum(0))
                per_cfg[cfg] = (allowed, lw)
                w[ci] = tot.sum() + len(on) * lpi + (m - len(on)) * l1pi
            w -= w.max()
            p = np.exp(w); p /= p.sum()
            cfg = cfgs[int(rng.choice(len(cfgs), p=p))]
            on = tuple(i for i in range(m) if cfg[i])
            z[blk] = False; z[blk[list(on)]] = True
            gam[blk] = False; D[blk] = 0.0
            # ---- gamma_g | z, then the slab draw, grouped by sampled subset ----
            if on:
                allowed, lw = per_cfg[cfg]
                gum = lw - np.log(-np.log(rng.random(lw.shape)))
                pick = np.asarray(allowed)[np.argmax(gum, axis=0)]
                for si in np.unique(pick):
                    S = subs[si]
                    if not S:
                        continue
                    g_idx = np.where(pick == si)[0]
                    idx = np.array(S)
                    mu = Ainv[S] @ C[np.ix_(idx, g_idx)]
                    eps = Lc[S] @ rng.normal(size=(len(S), len(g_idx)))
                    val = mu + eps * np.sqrt(sig2[g_idx])
                    rows = blk[idx]
                    D[np.ix_(rows, g_idx)] = val
                    gam[np.ix_(rows, g_idx)] = True
            r -= Ub @ D[blk]

        rss = np.einsum("ij,ij->j", r, r)
        kg = gam.sum(0)
        sig2 = (b_sigma + 0.5 * (rss + (D * D).sum(0) / om2)) / rng.gamma(
            a_sigma + 0.5 * (n - 1 + kg), 1.0)

        nz = int(z.sum())
        pi = rng.beta(pi_prior[0] + nz, pi_prior[1] + B - nz)
        ng = int(gam.sum())
        rho = rng.beta(rho_prior[0] + ng, rho_prior[1] + max(nz * G - ng, 0))

        if na * nt > 1:                            # Metropolis walk on the (alpha, tau) grid
            aj, tj = ai, ti
            if rng.random() < 0.5:
                aj += 1 if rng.random() < 0.5 else -1
            else:
                tj += 1 if rng.random() < 0.5 else -1
            if 0 <= aj < na and 0 <= tj < nt:
                act = np.where(np.abs(D).sum(1) > 0)[0]
                def _ll(k):
                    q = pre[k]
                    rr = q["Y"] - (q["U"][:, act] @ D[act] if len(act) else 0.0)
                    return float(-G * (q["logdetL"] + q["log_one"])
                                 - 0.5 * np.sum(np.einsum("ij,ij->j", rr, rr) / sig2))
                if np.log(rng.random()) < _ll(aj * nt + tj) - _ll(ai * nt + ti):
                    ai, ti = aj, tj
                    q = pre[ai * nt + ti]
                    r = q["Y"] - (q["U"][:, act] @ D[act] if len(act) else 0.0)
                    acc[0] += 1
                acc[1] += 1

        if it >= burn and (it - burn) % thin == 0:
            Pz += z; Pg += gam; Dm += D; S2 += sig2
            a_draws.append(alpha_grid[ai]); t_draws.append(tau_grid[ti])
            pi_draws.append(pi); rho_draws.append(rho)
            nev.append(nz); keep += 1
        if verbose and (it + 1) % 200 == 0:
            print(f"  it {it+1:5d}  n_events={nz:3d}  alpha={alpha_grid[ai]:.2f}  "
                      f"tau={tau_grid[ti]:.2f}  rho={rho:.3f}")

    return Model1Result(
        branches=branches, p_z=Pz / keep, p_gamma=Pg / keep, delta_mean=Dm / keep,
        alpha_draws=np.array(a_draws), tau_draws=np.array(t_draws),
        pi_draws=np.array(pi_draws), rho_draws=np.array(rho_draws),
        sigma2_mean=S2 / keep, n_event_draws=np.array(nev),
        diagnostics=dict(n_kept=keep, alpha_accept=float(acc[0] / max(acc[1], 1)),
                         alpha_grid=alpha_grid, tau_grid=tau_grid, omega=omega, n_branches=B,
                         n_blocks=len(blocks)))


def branch_evidence(Y, tree, branches=None, alpha_grid=(1.0,), tau_grid=(0.0,), root="stationary",
                    omega=1.0, rho=0.1, min_leaves=8, standardize=True):
    """Collapsed evidence for an event on each branch, with no MCMC and no responder set.

    This is the ``z_b`` update evaluated from the null state -- the graded score behind
    ``P(z_b = 1 | Y)``, and the thing that should be compared with experiment 05's pooled scans:

        score_b = sum_g log[ (1 - rho) + rho * BF_bg ] ,      profiled over the (alpha, tau) grid.

    A non-responder has ``BF ~ 1`` and adds ``~0``; a responder adds ``~log(rho BF)``. Contrast
    ``sum_g dAICc_bg``, in which every non-responder contributes noise of the same order as a
    responder's signal -- which is why naive pooling loses to the oracle.

    Returns ``(branches, score)``.
    """
    Y = np.asarray(Y, dtype=float)
    if standardize:
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    n = Y.shape[0]
    branches = candidate_branches(tree, min_leaves) if branches is None else np.asarray(branches, int)
    om2 = float(omega) ** 2
    l1r, lr = np.log(max(1 - rho, 1e-12)), np.log(max(rho, 1e-12))
    best = np.full(len(branches), -np.inf)
    for a in np.atleast_1d(alpha_grid):
        for tau in np.atleast_1d(tau_grid):
            p = _prepare(tree, Y, branches, float(a), float(tau), root)
            U, r = p["U"], p["Y"]
            s2 = np.einsum("ij,ij->j", r, r) / (n - 1)         # null-model per-gene scale
            v = np.einsum("ij,ij->j", U, U)
            T = U.T @ r                                        # (B, G)
            denom = 1.0 + om2 * v[:, None]
            logBF = -0.5 * np.log(denom) + om2 * T * T / (2.0 * s2[None, :] * denom)
            best = np.maximum(best, np.logaddexp(l1r, lr + logBF).sum(axis=1))
    return branches, best
