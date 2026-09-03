"""Hierarchical Bayesian mode detection.

Classifying each gene's *mode of evolution* on a lineage tree -- neutral drift, stabilising
selection, or adaptation to a shifted optimum -- has until now been done one gene at a time, by
fitting each hypothesis and picking a winner with an information criterion. That has a hard
information floor: a gene whose individual evidence is weak gets a confident answer that is barely
better than chance on small trees.

This subpackage shares structure across genes instead. Adaptive events are latent and live on
*branches*, so hundreds of genes each contributing weak evidence can localise the same event, while
a per-gene indicator decides which genes actually respond to it:

    z_b      ~ Bernoulli(pi)                            which BRANCHES carry an event
    g_bg|z_b ~ Bernoulli(z_b * rho)                     which GENES respond to it
    d_bg|g   ~ N(0, omega^2 sigma_g^2)  if g_bg else 0  the optimum shift
    Y_g      ~ N(theta0_g 1 + U(alpha) d_g,  sigma_g^2 [R(alpha) + tau I])

The read-out is a posterior over event locations, ``P(z_b = 1 | Y)``, together with the responding
gene set ``P(gamma_bg = 1 | Y)`` -- a discovery output that per-gene model selection cannot produce,
because it never asks where an event was.

Use it through :func:`scphytr.tl.detect_modes`. :func:`~scphytr.modes.baseline.classify_genes` keeps
the per-gene minimum-AICc rule available as the comparator it is.

Imported from the ``sparseou`` subproject; see ``analysis/scout/notes/`` for its development and
``analysis/scout/`` for the SCOUT comparison it was built against.
"""
from ._tree import Tree, parse_newick
from ._adapt import to_array_tree, leaf_order
from ._ou import node_optima, tip_mean, tip_cov, loglik
from .design import shift_design, candidate_branches
from .model import Model1Result, fit_model1, branch_evidence
from .simulate import SimData, simulate_dataset, simulate_tips, draw_events, leaf_regimes
from .baseline import paint_regimes, fit_models, classify_genes

__all__ = [
    "Tree", "parse_newick", "to_array_tree", "leaf_order",
    "node_optima", "tip_mean", "tip_cov", "loglik",
    "shift_design", "candidate_branches",
    "Model1Result", "fit_model1", "branch_evidence",
    "SimData", "simulate_dataset", "simulate_tips", "draw_events", "leaf_regimes",
    "paint_regimes", "fit_models", "classify_genes",
]
