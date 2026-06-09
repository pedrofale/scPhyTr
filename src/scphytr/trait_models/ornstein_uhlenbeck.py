import numpy as np

from .base import BaseTraitModel
from ..utils.pruning import ou_pruning_logpdf


class OrnsteinUhlenbeck(BaseTraitModel):
    """Multivariate Ornstein-Uhlenbeck trait model with a scalar mean-reversion.

    The trait vector evolves toward an optimum ``theta`` at a single rate
    ``alpha`` shared across traits, with diffusion covariance ``trait_cov_matrix``
    (K). Because alpha is scalar, the conditional covariance over any branch is a
    scalar multiple of K, preserving the K ⊗ C structure of the BM model and
    keeping the pruning likelihood O(n) over the tree.

    Parameters
    ----------
    tree : scphytr.utils.tree.Tree
    alpha : float
        Mean-reversion rate (alpha > 0). As alpha -> 0 the model tends to BM.
    theta : array-like, shape (p,)
        Trait optima (the OU pull target).
    trait_cov_matrix : array-like, shape (p, p)
        Diffusion (rate) covariance K.
    root_value : array-like, shape (p,), optional
        Fixed ancestral state above the root. Defaults to ``theta``.

    Notes
    -----
    A per-trait alpha (full drift matrix) breaks the scalar-covariance structure
    and would require p-variate messages; that is a planned extension.
    """

    def __init__(self, tree, alpha, theta, trait_cov_matrix, root_value=None,
                 learnable_parameters=('alpha', 'theta', 'rates')):
        super().__init__(tree, list(learnable_parameters))
        self.alpha = float(alpha)
        self.theta = np.asarray(theta, dtype=float).ravel()
        self.trait_cov_matrix = np.asarray(trait_cov_matrix, dtype=float)
        self.root_value = None if root_value is None else np.asarray(root_value, dtype=float).ravel()

    def score_pruning(self, alpha=None, theta=None, trait_cov_matrix=None, root_value="__default__"):
        """Linear-time OU marginal log-likelihood via Felsenstein's pruning."""
        alpha = self.alpha if alpha is None else alpha
        theta = self.theta if theta is None else np.asarray(theta, dtype=float).ravel()
        K = self.trait_cov_matrix if trait_cov_matrix is None else np.asarray(trait_cov_matrix, dtype=float)
        root = self.root_value if root_value == "__default__" else root_value
        return ou_pruning_logpdf(self.tree, alpha, theta, K, root_value=root)

    def set_trait_cov_matrix(self, rates):
        self.trait_cov_matrix = np.diag(np.asarray(rates, dtype=float).ravel())
