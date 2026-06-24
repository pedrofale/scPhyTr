from ete3 import PhyloTree
import numpy as np
import pandas as pd
import itertools
from jax.scipy.linalg import cholesky
import jax.numpy as jnp

class Tree(object):
    """
    A class to represent a phylogenetic tree using ete3 and a dictionary for efficient traversal.
    """
    def __init__(self, tree_file=None):
        self.phylotree = None
        self.root = None
        self.trait_values = None # Species x traits matrix
        self.trait_values_traitmajor = None # Trait-major order: [t1(s1..sn), t2(s1..sn), ...]
        self.species_cov_matrix = None
        self.species_cholesky = None
        if tree_file is not None:
            self.load_newick(tree_file)

    def simulate_tree(self, n_leaves, n_internal_nodes, n_branches, branch_length_mean, branch_length_std, trait_mean, trait_std):
        self.phylotree = PhyloTree.simulate_tree(n_leaves, n_internal_nodes, n_branches, branch_length_mean, branch_length_std)
        self.root = self.phylotree.get_tree_root()
        self.make_species_cov_matrix()

    def get_species_cov_matrix(self):
        if self.species_cov_matrix is None:
            self.make_species_cov_matrix()
        return self.species_cov_matrix

    def get_species_cholesky(self):
        if self.species_cholesky is None:
            self.make_species_cov_matrix()
        return self.species_cholesky

    def make_species_cov_matrix(self):
        # Create covariance matrix for species from tree -- TODO: benchmark this
        species_cov_matrix = pd.DataFrame(index=self.phylotree.get_leaf_names(), columns=self.phylotree.get_leaf_names())
        species_cov_matrix.index = species_cov_matrix.columns
        
        def descend(root, total_length=0):
            clades = []
            for child in root.children:
                clades.append(descend(child, total_length=total_length + root.dist))
            if root.is_leaf():
                species_cov_matrix.loc[root.name, root.name] = total_length + root.dist
                return root.name
            else:
                for c1, c2 in itertools.combinations(clades, 2):
                    a = np.array(c1, dtype=object)
                    b = np.array(c2, dtype=object)
                    # Broadcasting to generate all combinations without explicit Python loops
                    pairs = np.stack(np.meshgrid(a, b), -1).reshape(-1, 2)
                    for pair in pairs:
                        species_cov_matrix.loc[pair[0], pair[1]] = total_length + root.dist
            return clades

        descend(self.root, total_length=0)
        species_cov_matrix = species_cov_matrix.combine_first(species_cov_matrix.T)
        self.species_cov_matrix = species_cov_matrix.astype(float)
        self.species_cholesky = jnp.asarray(cholesky(self.species_cov_matrix.values, lower=True))

    def get_trait_values(self):
        if self.trait_values is None:
            self.make_trait_values()
        return self.trait_values

    def get_trait_values_traitmajor(self):
        if self.trait_values_traitmajor is None:
            self.make_trait_values()
        return self.trait_values_traitmajor

    def set_trait_values(self, species_trait_values):
        """
        species_trait_values: dictionary with species as keys and trait values as dictionaries with trait names as keys
        """
        for node in self.phylotree.get_leaves():
            node.trait = {character: species_trait_values[node.name][character] for character in species_trait_values[node.name]}
        self.make_trait_values()

    def make_trait_values(self):
        trait_values = []
        for leaf in self.phylotree.get_leaves():
            if isinstance(leaf.trait, dict):
                traits = []
                for trait in leaf.trait:
                    traits.append(leaf.trait[trait])
                trait_values.append(traits)
            else:
                trait_values.append(leaf.trait)
        self.trait_values = np.array(trait_values)
        self.trait_values_traitmajor = self.trait_values.reshape(-1, order='F')

    def get_trait_names(self):
        return list(self.phylotree.get_leaves()[0].trait.keys())

    def load_newick(self, tree_file):
        self.phylotree = PhyloTree(tree_file)
        self.root = self.phylotree.get_tree_root()
        # The dense species covariance is a validation-only oracle and does not
        # support multifurcating (polytomous) trees; build it lazily via the
        # getters so the O(n) pruning path works on any tree topology.
    
    def save_newick(self, tree_file):
        self.phylotree.write(format=1, outfile=tree_file)