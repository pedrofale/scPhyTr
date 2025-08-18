import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

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
        return multivariate_normal.logpdf(self.tree.get_trait_values(), a, V)

    def sample_parameters(self):
        return self.trait_means.values.ravel(), np.eye(self.trait_cov_matrix.shape[0])

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

    # Our custom proposal distribution
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