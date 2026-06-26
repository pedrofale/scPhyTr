"""Multi-rate Brownian motion: the diffusion rate sigma^2 changes between painted regimes.

O'Meara's BMS model --- the trait model behind clade-specific *rate* shifts. The per-branch
variance is ``rates[regime(edge)] * branch_length``; the regimes are a painting on the tree
(see :func:`scphytr.utils.pruning.paint_regimes`). Used with the ``Laplace`` inference's exact
pruning path (directly observed / Gaussian trait).
"""
import numpy as np

from .base import BaseTraitModel


class MultiRateBM(BaseTraitModel):
    def __init__(self, tree, rates, regimes, n_regimes, root_value=None,
                 learnable_parameters=("rates",)):
        super().__init__(tree, list(learnable_parameters))
        self.rates = np.asarray(rates, dtype=float).ravel()
        self.regimes = regimes
        self.n_regimes = int(n_regimes)
        self.root_value = None if root_value is None else float(root_value)

    def process_params(self):
        return dict(alpha=None, theta=None, sigma2=None, regimes=self.regimes,
                    n_regimes=self.n_regimes, root_value=self.root_value, rates=self.rates)

    def pack(self):
        return np.log(np.maximum(self.rates, 1e-9))

    def unpack(self, x):
        self.rates = np.exp(np.asarray(x, dtype=float))
