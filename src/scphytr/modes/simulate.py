"""Minimal simulator for the hierarchical sparse-OU model.

Generative model actually sampled here (the "scientific" nested spike-and-slab, with the discrete
indicators drawn explicitly -- no shrinkage approximation):

    z_b     ~ Bernoulli(pi)                      few branches carry an adaptive event
    g_bg    ~ Bernoulli(rho)  if z_b = 1         few genes respond to it
    d_bg    ~ N(0, omega^2)   if z_b g_bg = 1    the optimum shift, else exactly 0
    theta_b = theta_{parent(b)} + d_b            optima inherited down the tree
    X       propagated by the OU transition, tips observed (optionally with noise)

Tips are simulated FORWARD branch by branch, deliberately independent of the analytic mean/covariance
in :mod:`scphytr.modes._ou` -- so agreement between the two is a real check, not a tautology.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._ou import node_optima, ou_decay, ou_var

__all__ = ["SimData", "draw_events", "simulate_tips", "simulate_dataset", "leaf_regimes"]


@dataclass
class SimData:
    """Simulated dataset plus the ground truth needed to score recovery."""
    Y: np.ndarray                       # (n_leaves, G) observed tip expression
    tree: object
    alpha: float
    sigma: float
    theta0: np.ndarray                  # (G,)
    delta: np.ndarray                   # (n_nodes, G) true optimum shifts
    z: np.ndarray                       # (n_nodes,) bool, true event branches
    gamma: np.ndarray                   # (n_nodes, G) bool, true gene participation
    event_branches: np.ndarray          # indices of true event branches
    params: dict = field(default_factory=dict)

    @property
    def n_leaves(self):
        return self.Y.shape[0]

    @property
    def n_genes(self):
        return self.Y.shape[1]

    def __repr__(self):
        return (f"SimData(n_leaves={self.n_leaves}, n_genes={self.n_genes}, "
                f"events={list(self.event_branches)}, alpha={self.alpha}, sigma={self.sigma})")


def _candidate_branches(tree, min_clade=4, max_clade_frac=0.75, exclude_root_children=False):
    """Branches eligible to carry an event: not the root, and subtending a clade of usable size."""
    sets = tree.leaf_sets()
    n = tree.n_leaves
    out = []
    for v in range(tree.n_nodes):
        if tree.parent[v] < 0:
            continue
        if exclude_root_children and tree.parent[v] == 0:
            continue
        k = len(sets[v])
        if k >= min_clade and k <= max_clade_frac * n:
            out.append(v)
    return np.array(out, dtype=int)


def draw_events(tree, n_genes, n_events=3, frac_genes=0.2, omega=1.0, rng=None,
                min_clade=4, max_clade_frac=0.75, branches=None, min_separation=0):
    """Draw (z, gamma, delta). ``frac_genes`` is rho, the fraction of genes responding per event."""
    rng = np.random.default_rng() if rng is None else rng
    z = np.zeros(tree.n_nodes, dtype=bool)
    gamma = np.zeros((tree.n_nodes, n_genes), dtype=bool)
    delta = np.zeros((tree.n_nodes, n_genes))

    if branches is None:
        cand = _candidate_branches(tree, min_clade=min_clade, max_clade_frac=max_clade_frac)
        if len(cand) < n_events:
            raise ValueError(f"only {len(cand)} candidate branches for {n_events} events")
        chosen = []
        pool = list(cand)
        for _ in range(n_events):
            if not pool:
                raise ValueError("ran out of candidate branches under the separation constraint")
            b = int(rng.choice(pool))
            chosen.append(b)
            if min_separation > 0:                      # drop ancestors/descendants nearby
                banned = _neighbourhood(tree, b, min_separation)
                pool = [x for x in pool if x not in banned]
            else:
                pool = [x for x in pool if x != b]
        chosen = np.array(sorted(chosen), dtype=int)
    else:
        chosen = np.asarray(branches, dtype=int)

    for b in chosen:
        z[b] = True
        k = max(1, int(round(frac_genes * n_genes)))
        which = rng.choice(n_genes, size=k, replace=False)
        gamma[b, which] = True
        delta[b, which] = rng.normal(0.0, omega, size=k)
    return z, gamma, delta, chosen


def _neighbourhood(tree, b, radius):
    """Branch indices within ``radius`` steps of ``b`` along ancestor/descendant paths."""
    out = {int(b)}
    v, r = b, radius
    while r > 0 and tree.parent[v] >= 0:
        v = int(tree.parent[v]); out.add(v); r -= 1
    stack = [(int(b), radius)]
    while stack:
        u, r = stack.pop()
        if r <= 0:
            continue
        for c in tree.children[u]:
            out.add(int(c)); stack.append((int(c), r - 1))
    return out


def simulate_tips(tree, alpha, sigma, theta0, delta, rng=None, root="stationary",
                  sigma_obs=0.0, x_root=None):
    """Forward-simulate the OU process down the tree; return tip values (n_leaves, G)."""
    rng = np.random.default_rng() if rng is None else rng
    th = node_optima(tree, theta0, delta)
    G = th.shape[1]
    sigma2 = float(sigma) ** 2
    X = np.zeros((tree.n_nodes, G))
    if x_root is not None:
        X[0] = np.asarray(x_root, dtype=float)
    elif root == "stationary" and alpha > 0:
        X[0] = th[0] + rng.normal(0.0, np.sqrt(sigma2 / (2.0 * alpha)), size=G)
    else:
        X[0] = th[0]
    for v in tree.preorder:
        p = tree.parent[v]
        if p < 0:
            continue
        phi = ou_decay(alpha, tree.dist[v])
        sd = np.sqrt(max(ou_var(alpha, sigma2, tree.dist[v]), 0.0))
        X[v] = phi * X[p] + (1.0 - phi) * th[v] + rng.normal(0.0, sd, size=G)
    Y = X[tree.leaves]
    if sigma_obs > 0:
        Y = Y + rng.normal(0.0, sigma_obs, size=Y.shape)
    return Y


def simulate_dataset(tree, n_genes=500, alpha=0.75, sigma=0.75, n_events=3, frac_genes=0.2,
                     omega=1.5, theta0_sd=1.0, sigma_obs=0.0, root="stationary", seed=0,
                     min_clade=4, max_clade_frac=0.75, branches=None, min_separation=0):
    """One end-to-end simulated dataset with ground truth attached."""
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0.0, theta0_sd, size=n_genes)
    z, gamma, delta, chosen = draw_events(
        tree, n_genes, n_events=n_events, frac_genes=frac_genes, omega=omega, rng=rng,
        min_clade=min_clade, max_clade_frac=max_clade_frac, branches=branches,
        min_separation=min_separation)
    Y = simulate_tips(tree, alpha, sigma, theta0, delta, rng=rng, root=root, sigma_obs=sigma_obs)
    return SimData(Y=Y, tree=tree, alpha=alpha, sigma=sigma, theta0=theta0, delta=delta,
                   z=z, gamma=gamma, event_branches=chosen,
                   params=dict(n_genes=n_genes, frac_genes=frac_genes, omega=omega,
                               sigma_obs=sigma_obs, root=root, seed=seed))


def leaf_regimes(tree, z):
    """Leaf regime labels implied by the event branches -- the input SCOUT would need.

    A leaf's regime is the set of event branches on its root-to-leaf path, so this is exactly the
    'adaptive landscape' a SCOUT-style method must be GIVEN (and which our model instead infers).
    """
    labels = {}
    tags = [""] * tree.n_nodes
    for v in tree.preorder:
        p = tree.parent[v]
        base = "" if p < 0 else tags[p]
        tags[v] = base + (f"|{v}" if z[v] else "")
    uniq = {}
    for i, l in enumerate(tree.leaves):
        t = tags[l] or "root"
        if t not in uniq:
            uniq[t] = f"R{len(uniq)}"
        labels[i] = uniq[t]
    return np.array([labels[i] for i in range(tree.n_leaves)])
