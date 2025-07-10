from ete3 import PhyloTree
import numpy as np
import pandas as pd


class Tree(object):
    """
    A class to represent a phylogenetic tree using ete3 and a dictionary for efficient traversal.
    """
    def __init__(self, tree_file=None):
        self.phylotree = None
        self.root = None
        if tree_file is not None:
            self.load_newick(tree_file)

    def make_root(self):
        pass

    def simulate_tree(self, n_leaves, n_internal_nodes, n_branches, branch_length_mean, branch_length_std, trait_mean, trait_std):
        self.phylotree = PhyloTree.simulate_tree(n_leaves, n_internal_nodes, n_branches, branch_length_mean, branch_length_std)
        self.root = self.make_root()

    def get_species_cov_matrix(self):
        # Create covariance matrix for species from tree
        species_cov_matrix = pd.DataFrame(np.zeros((len(self.phylotree.get_leaf_names()), len(self.phylotree.get_leaf_names()))))
        species_cov_matrix.columns = self.phylotree.get_leaf_names()
        species_cov_matrix.index = self.phylotree.get_leaf_names()
        def descend(root, total_length=0, visited=[]):
            if root.is_leaf():
                species_cov_matrix[root.name, root.name] = total_length
                species_cov_matrix[root.name, root.name] = total_length

            for child in root.children:
                child_names = descend(child, total_length=total_length+rootdist)
                species_cov_matrix[root.name, child_names] = 
            
            species_cov_matrix[root.name, visited] = total_length
        descend(self.root, total_length=0)
        return species_cov_matrix


    def load_newick(self, tree_file):
        self.phylotree = PhyloTree(tree_file)
        self.root = self.make_root()
    
    def save_newick(self, tree_file):
        self.phylotree.write(format=1, outfile=tree_file)