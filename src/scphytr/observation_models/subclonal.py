"""Subclonal observation model: the cells of a subclone as replicate observations.

Each leaf of the tree is a subclone (or, on a single-cell tree, a single cell); its cells are
modelled as *repeated* Poisson (or negative-binomial) draws of that leaf's latent log-rate —
the cells are **never collapsed to pseudobulk**. This wraps the validated cells-as-replicates
engine (:class:`scphytr.inference.laplace.MultiCellPoissonObservation`) so it plugs into both
the modular observation-model registry and the ``Laplace`` inference (it already exposes the
``mode_init/loglik/grad/neg_hess_diag`` leaf-likelihood contract).

``dispersion=None`` -> Poisson; ``dispersion=r`` -> negative-binomial (within-clone plasticity).
"""
from ..inference.laplace import MultiCellPoissonObservation
from .base import BaseObservationModel


class SubclonalObservation(MultiCellPoissonObservation, BaseObservationModel):
    def __init__(self, counts, offsets, leaf_index, n_leaves, dispersion=None,
                 univariate=True, learnable_parameters=None):
        MultiCellPoissonObservation.__init__(
            self, counts, offsets, leaf_index, n_leaves,
            dispersion=dispersion, univariate=univariate)
        if learnable_parameters is None:
            learnable_parameters = ["dispersion"] if dispersion is not None else []
        self.learnable_parameters = list(learnable_parameters)

    def get_learnable_parameters(self):
        return self.learnable_parameters


def NegativeBinomial(counts, offsets, leaf_index, n_leaves, dispersion=10.0, **kw):
    """Subclonal observation with within-clone negative-binomial overdispersion."""
    return SubclonalObservation(counts, offsets, leaf_index, n_leaves,
                                dispersion=dispersion, **kw)
