"""The linear-model view of a per-branch-optimum OU process.

The single structural fact that makes a Bayesian treatment cheap: **the optima enter only the
mean, and the mean is LINEAR in the shifts**. Writing theta_b = theta_parent(b) + delta_b and
starting the root at its own optimum,

    E[Y_g] = theta0_g * 1  +  U(alpha) delta_g ,        Cov[Y_g] = sigma_g^2 R(alpha)

where ``U(alpha)[i, b]`` is the OU-weighted exposure of tip *i* to an optimum shift on branch *b*
(zero unless *b* is on the root-to-*i* path). So, given alpha, the whole problem is Gaussian linear
regression with a shared design matrix and a structured-sparse coefficient matrix -- i.e. Bayesian
variable selection, not a bespoke phylogenetic optimiser.

Note ``U``'s columns for two branches in an unbroken chain are NOT identical (they differ by
exposure time), even though they induce the same leaf partition. That is exactly the degeneracy
SCOUT's OUx suffers from, and the reason a prior on delta partially breaks it.
"""
from __future__ import annotations

import numpy as np

from ._ou import ou_decay

__all__ = ["shift_design", "candidate_branches"]


def shift_design(tree, alpha, branches=None):
    """Tip exposure matrix ``U`` (n_leaves, n_branches) such that ``E[Y] = theta0 + U delta``.

    Computed by propagating the tip-mean recursion for all unit shifts at once:
    ``m_v = phi_v m_parent + (1 - phi_v) T_v`` with ``T[v, b] = 1{b is an ancestor-or-self of v}``.
    """
    branches = np.arange(tree.n_nodes) if branches is None else np.asarray(branches, dtype=int)
    B = len(branches)
    col = -np.ones(tree.n_nodes, dtype=int)
    col[branches] = np.arange(B)

    T = np.zeros((tree.n_nodes, B))      # optimum under a unit shift on each branch
    M = np.zeros((tree.n_nodes, B))      # expected trait
    for v in tree.preorder:
        p = tree.parent[v]
        if p < 0:
            continue
        T[v] = T[p]
        if col[v] >= 0:
            T[v, col[v]] += 1.0
        phi = ou_decay(alpha, tree.dist[v])
        M[v] = phi * M[p] + (1.0 - phi) * T[v]
    return M[tree.leaves]


def candidate_branches(tree, min_leaves=8):
    """Branches eligible to carry an event: enough tips inside and outside to be estimable."""
    sets = tree.leaf_sets()
    return np.array([v for v in range(tree.n_nodes)
                     if tree.parent[v] >= 0
                     and min_leaves <= len(sets[v]) <= tree.n_leaves - min_leaves], dtype=int)
