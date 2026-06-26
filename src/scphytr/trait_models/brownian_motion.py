import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow.linalg as tfl

from .base import BaseTraitModel
from ..utils.jax_utils import build_cholesky, mvn_kron_logpdf_traitmajor_with_L, sample_mvnormal_kron_traitmajor


class BrownianMotion(BaseTraitModel):
    def __init__(self, tree, trait_means, trait_cov_matrix, learnable_parameters=['rates']):
        super().__init__(tree, learnable_parameters)
        """
        tree: ete3 tree
        trait_means: means of traits, named
        trait_cov_matrix: covariances between traits, named -- TODO: extend to per-clade trait covariance matrices
        Expanded multivariate Brownian motion model will do trait-major order, i.e. species1_trait1, species2_trait1, species1_trait2, species2_trait2
        For learning, uses Cholesky decomposition of trait covariance matrix.
        """
        self.trait_means = trait_means # means of traits
        self.trait_cov_matrix = trait_cov_matrix # correlations between traits -- TODO: extend to per-clade trait covariance matrices
        self.validate_trait_shapes()

    def reshape_trait_values(self, trait_values):
        """
        Reshape trait values from species x traits to trait-major order.
        """
        return trait_values.reshape(-1, order='F') # species x traits to trait-major order

    def validate_trait_shapes(self):
        if self.trait_means.shape[0] != self.trait_cov_matrix.shape[0]:
            raise ValueError(f"Trait means and covariance matrix must have the same number of traits. Got {self.trait_means.shape[0]} and {self.trait_cov_matrix.shape[0]}")
        n = self.tree.phylotree.get_leaves()[0]
        if 'trait' in n.features:
            if self.trait_means.shape[0] != len(n.trait):
                raise ValueError(f"Trait means and covariance matrix must have the same number of traits as the nodes. Got {self.trait_means.shape[0]} and {len(n.trait)}")
    @staticmethod
    def multivariate_brownian_motion_path(T, N, cov_matrix):
        """
        Generate a multivariate Brownian motion path.
        
        Parameters:
        T (float): Total time.
        N (int): Number of steps.
        cov_matrix (np.ndarray): Covariance matrix.
        
        Returns:
        np.ndarray: Brownian motion path.
        """
        dt = T / N  # Time step size
        dW = np.random.multivariate_normal(np.zeros(cov_matrix.shape[0]), cov_matrix*dt, N)  # Increments for every trait
        W = np.cumsum(dW, axis=0)  # Cumulative sum to get the path
        return W

    def simulate_paths(self, seed=42, N=100):
        """
        Simulate paths for all species in the tree.
        Parameters:
        seed (int): Seed for random number generator.
        N (int): Number of steps per branch?.

        Returns:
        np.ndarray: Array of species trait values.
        """
        np.random.seed(seed)
        species_paths = dict()
        def descend(root, path):
            if root.is_leaf():
                species_paths[root.name] = path
            for child in root.children:
                local_path = self.multivariate_brownian_motion_path(child.dist, int(N*child.dist), self.trait_cov_matrix)
                new_path = np.concatenate((path, path[-1] + local_path))
                descend(child, new_path)

        path = self.trait_means.values.ravel() + self.multivariate_brownian_motion_path(self.tree.root.dist, int(N*self.tree.root.dist), self.trait_cov_matrix)
        descend(self.tree.root, path)        
        return species_paths

    def simulate_traits(self, seed=42):
        np.random.seed(seed)
        # Create variance-covariance matrix
        a = np.repeat(self.trait_means, self.n_species) 
        V = np.kron(self.trait_cov_matrix, self.tree.get_species_cov_matrix()) 
        species_trait_values = np.random.multivariate_normal(a, V) 
        species_trait_values = species_trait_values.reshape(self.n_species, -1, order='F')
        return pd.DataFrame(species_trait_values, index=self.tree.get_species_cov_matrix().index, columns=self.trait_means.index)
        
    def set_trait_cov_matrix(self, rates):
        self.trait_cov_matrix = np.diag(rates)

    def compute_analytical_solution(self, unbiased=True):
        """
        Compute the analytical solution for the trait means and rates.
        If unbiased=True, the rates are divided by (n-1) to getthe REML esrtimate.
        """
        # Compute the analytical solution for the trait means
        n = self.n_species
        C = self.tree.get_species_cov_matrix()
        inv_C = np.linalg.inv(C)
        x = self.tree.get_trait_values()
        ones = np.ones(n)[:, np.newaxis] # row vector of ones
        trait_means = (np.linalg.inv(ones.T @ inv_C @ ones) @ (ones.T @ inv_C @ x)).T
        trait_cov_matrix = (x - ones @ trait_means.T).T @ inv_C @ (x - ones @ trait_means.T) 
        if unbiased:
            trait_cov_matrix = trait_cov_matrix / (n-1)
        else:
            trait_cov_matrix = trait_cov_matrix / n
        return trait_means, trait_cov_matrix

    def score(self, trait_means, trait_cov_matrix):
        a = np.repeat(trait_means, self.n_species)  # trait-major order: species1_trait1, species2_trait1, species1_trait2, species2_trait2
        V = np.kron(trait_cov_matrix, self.tree.get_species_cov_matrix()) 
        y = self.reshape_trait_values(self.tree.get_trait_values())
        return multivariate_normal.logpdf(y, a, V)

    def score_pruning(self, trait_means, trait_cov_matrix):
        """Linear-time equivalent of ``score`` via Felsenstein's pruning.

        Computes the same homogeneous-BM marginal log-likelihood as ``score`` but
        in O(n) over the tree, without forming the dense K ⊗ C covariance.
        """
        from ..utils.pruning import bm_pruning_logpdf
        return bm_pruning_logpdf(self.tree, np.asarray(trait_means).ravel(), trait_cov_matrix)

    def sample_parameters(self):
        free_cholesky = np.linalg.cholesky(self.trait_cov_matrix)
        free_cholesky = np.tril(free_cholesky, -1) + np.diag(np.log(np.diag(free_cholesky)))
        return self.trait_means.values.ravel(), free_cholesky # optimize Cholesky

    def set_parameters(self, params):
        self.trait_means = pd.Series(params[0], index=self.trait_means.index)
        L_params = params[1]
        Lk = build_cholesky(L_params)
        self.trait_cov_matrix = pd.DataFrame(Lk @ Lk.T, index=self.trait_cov_matrix.index, columns=self.trait_cov_matrix.columns)

    def _species_cov(self):
        # ensure JAX array
        return jnp.asarray(self.tree.get_species_cov_matrix().values)

    def _species_cholesky(self):
        # ensure JAX array
        return self.tree.get_species_cholesky()

    def sample_prior(self, params, rng):
        trait_means, trait_cov = params                       # jnp arrays
        return tfd.MultivariateNormalFullCovariance(loc=trait_means, covariance_matrix=trait_cov).sample(seed=rng)

    def logpdf_prior(self, trait_values, params):
        # trait_means, trait_cov = params                       # jnp arrays
        # a = jnp.repeat(trait_means, self.n_species)             # trait-major order
        # V = jnp.kron(trait_cov, self._species_cov())          # (p*n)×(p*n)
        # return tfd.MultivariateNormalFullCovariance(loc=a, covariance_matrix=V).log_prob(trait_values)
        trait_means, L_params = params             # <- optimize L_params, not K
        Lk = build_cholesky(L_params)              # trait Cholesky
        z = jnp.asarray(trait_values)
        # a = jnp.repeat(trait_means, self.n_species)             # trait-major order
        # V = jnp.kron(Lk @ Lk.T, self._species_cov())          # (p*n)×(p*n)
        # return tfd.MultivariateNormalFullCovariance(loc=a, covariance_matrix=V).log_prob(z)        
        return mvn_kron_logpdf_traitmajor_with_L(z, trait_means, Lk, self._species_cholesky(), self.n_species)        

    # Our custom proposal distribution for Importance Sampling
    # Same as prior
    def sample_proposal(self, params, rng):
        trait_means, L_params = params             # <- optimize L_params, not K
        Lk = build_cholesky(L_params)              # trait Cholesky
        return sample_mvnormal_kron_traitmajor(rng, self._species_cholesky(), Lk, trait_means)

    # Same as prior
    def logpdf_proposal(self, trait_values, params):
        trait_means, L_params = params             # <- optimize L_params, not K
        Lk = build_cholesky(L_params)              # trait Cholesky
        z = jnp.asarray(trait_values)
        return mvn_kron_logpdf_traitmajor_with_L(z, trait_means, Lk, self._species_cholesky(), self.n_species)   

    # # Our custom proposal distribution for Variational Inference
    # # Mean-field: nxp means, nxp variances, no correlations
    # def sample_variational_parameters(self):
    #     return jnp.zeros(self.trait_means.shape[0]*self.n_species), jnp.zeros(self.trait_means.shape[0]*self.n_species)
    #     # a = np.ones(self.trait_means.shape[0]*self.n_species) * 10.
    #     # a[self.n_species:] = -10.
    #     # v = np.zeros(self.trait_means.shape[0]*self.n_species)
    #     # v[:self.n_species] = jnp.log(.1)
    #     # v[self.n_species:] = jnp.log(1.)
    #     # return jnp.array(a), jnp.array(v)

    # def set_variational_parameters(self, variational_params):
    #     self.variational_params = variational_params

    # def sample_variational_proposal(self, variational_params, rng):
    #     trait_means, log_trait_variances = variational_params
    #     trait_variances = jnp.exp(log_trait_variances)
    #     return tfd.MultivariateNormalDiag(loc=trait_means, scale_diag=trait_variances).sample(seed=rng)

    # def logpdf_variational_proposal(self, trait_values, variational_params):
    #     trait_means, log_trait_variances = variational_params
    #     trait_variances = jnp.exp(log_trait_variances)
    #     return tfd.MultivariateNormalDiag(loc=trait_means, scale_diag=trait_variances).log_prob(trait_values)

    # def kl_divergence(self, trait_params, variational_params):
    #     # TODO: optimize this to avoid forming the Kronecker product
    #     trait_means, L_params = trait_params
    #     Lk = build_cholesky(L_params)
    #     a = jnp.repeat(trait_means, self.n_species)             # trait-major order
    #     V = jnp.kron(Lk @ Lk.T, self._species_cov())          # (p*n)×(p*n)
    #     variational_means, log_variational_variances = variational_params # p*n
    #     variational_variances = jnp.exp(log_variational_variances)
    #     return tfd.MultivariateNormalDiag(loc=variational_means, scale_diag=variational_variances).kl_divergence(tfd.MultivariateNormalFullCovariance(loc=a, covariance_matrix=V))

    # Structured VI: nxp means, pxp trait covariances (cholesky)
    def sample_variational_parameters(self):
        return jnp.zeros((self.n_species, self.trait_means.shape[0])), jnp.zeros((self.trait_means.shape[0], self.trait_means.shape[0]))
        # a = np.ones(self.trait_means.shape[0]*self.n_species) * 10.
        # a[self.n_species:] = -10.
        # v = np.zeros(self.trait_means.shape[0]*self.n_species)
        # v[:self.n_species] = jnp.log(.1)
        # v[self.n_species:] = jnp.log(1.)
        # return jnp.array(a), jnp.array(v)

    def set_variational_parameters(self, variational_params):
        self.variational_params = variational_params

    def sample_variational_proposal(self, variational_params, rng):
        variational_means, variational_cholesky = variational_params
        variational_cholesky = build_cholesky(variational_cholesky)
        Xi = jax.random.normal(rng, shape=(self.n_species, self.n_traits))

        # Z = M + Lc Xi Lq^T
        Z = variational_means + self._species_cholesky() @ Xi @ variational_cholesky.T
        return Z


    def logpdf_variational_proposal(self, trait_values, variational_params):
        variational_means, variational_cholesky = variational_params
        variational_cholesky = build_cholesky(variational_cholesky)

        M, variational_cholesky = variational_params
        Lq = build_cholesky(variational_cholesky)
        Sq = Lq @ Lq.T
        n, p = trait_values.shape

        diff = trait_values - M
        # Mahalanobis term: tr(Sq^{-1} diff^T C^{-1} diff)
        Sq_inv = jnp.linalg.inv(Sq)
        mahal = jnp.trace(Sq_inv @ (diff.T @ self._species_cov_inv() @ diff))

        logdet_Sq = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(Sq))))
        logdet_C  = 2.0 * jnp.sum(jnp.log(jnp.diag(self._species_cholesky())))

        logpdf = -0.5 * (n*p*jnp.log(2*jnp.pi) + n*logdet_Sq + p*logdet_C + mahal)
        return logpdf


        op = tfl.LinearOperatorKronecker([
            tfl.LinearOperatorLowerTriangular(variational_cholesky),   # column factor
            tfl.LinearOperatorLowerTriangular(self._species_cholesky()),   # row factor
        ])
        loc = jnp.reshape(variational_means, [-1])                    # vec(M)
        return tfd.MultivariateNormalLinearOperator(loc=loc, scale=op).log_prob(trait_values)

    def kl_divergence(self, trait_params, variational_params):
        # TODO: optimize this to avoid forming the Kronecker product
        trait_means, L_params = trait_params
        Lk = build_cholesky(L_params)
        a = jnp.repeat(trait_means, self.n_species)             # trait-major order
        V = jnp.kron(Lk @ Lk.T, self._species_cov())          # (p*n)×(p*n)
        variational_means, variational_cholesky = variational_params # p*n
        variational_cholesky = build_cholesky(variational_cholesky)
        op = tfl.LinearOperatorKronecker([
            tfl.LinearOperatorLowerTriangular(variational_cholesky),   # column factor
            tfl.LinearOperatorLowerTriangular(self._species_cholesky()),   # row factor
        ])
        loc = jnp.reshape(variational_means, [-1])                    # vec(M)
        return tfd.MultivariateNormalLinearOperator(loc=loc, scale=op).kl_divergence(tfd.MultivariateNormalFullCovariance(loc=a, covariance_matrix=V))

    # ----- modular Laplace contract (univariate) ----------------------------------
    @staticmethod
    def _scalar(a):
        a = a.values if hasattr(a, "values") else a
        return float(np.ravel(np.asarray(a, dtype=float))[0])

    def process_params(self):
        """(alpha=0 for BM, theta=root mean, sigma2, regimes, root_value) for the engine."""
        mu = self._scalar(self.trait_means)
        return dict(alpha=0.0, theta=mu, sigma2=self._scalar(self.trait_cov_matrix),
                    regimes=None, n_regimes=1, root_value=mu, rates=None)

    def pack(self):
        return np.array([self._scalar(self.trait_means),
                         np.log(max(self._scalar(self.trait_cov_matrix), 1e-9))], float)

    def unpack(self, x):
        self.trait_means = np.array([float(x[0])])
        self.trait_cov_matrix = np.array([[float(np.exp(x[1]))]])