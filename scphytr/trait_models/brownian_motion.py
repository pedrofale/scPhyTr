import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from .base import Base

class BrownianMotion(Base):
    def __init__(self, tree, trait_means, trait_cov_matrix, learnable_parameters=['rates']):
        super().__init__(tree, learnable_parameters)
        """
        tree: ete3 tree
        trait_means: means of traits, named
        trait_cov_matrix: correlations between traits, named -- TODO: extend to per-clade trait covariance matrices
        """
        self.trait_means = trait_means # means of traits
        self.trait_cov_matrix = trait_cov_matrix # correlations between traits -- TODO: extend to per-clade trait covariance matrices
        self.validate_trait_shapes()

    def validate_trait_shapes(self):
        if self.trait_means.shape[0] != self.trait_cov_matrix.shape[0]:
            raise ValueError(f"Trait means and covariance matrix must have the same number of traits. Got {self.trait_means.shape[0]} and {self.trait_cov_matrix.shape[0]}")
        n = self.tree.phylotree.get_leaves()[0]
        if 'trait' in n.features:
            if self.trait_means.shape[0] != len(n.trait):
                raise ValueError(f"Trait means and covariance matrix must have the same number of traits as the nodes. Got {self.trait_means.shape[0]} and {len(n.trait)}")

    def score(self, trait_means, trait_cov_matrix):
        a = np.repeat(trait_means, self.n_species)  # species1_trait1, species1_trait2, species2_trait1, species2_trait2, ...
        V = np.kron(trait_cov_matrix, self.tree.get_species_cov_matrix()) 
        return multivariate_normal.logpdf(self.tree.trait_values, a, V)

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

    def compute_analytical_solution(self):
        X = self.tree.get_trait_values()
        a = np.repeat(self.trait_means, self.n_species) 
        R = np.kron(self.trait_cov_matrix, self.tree.get_species_cov_matrix()) 
        return multivariate_normal.logpdf(X, a, R)