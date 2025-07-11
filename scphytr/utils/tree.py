from ete3 import PhyloTree
import numpy as np
import pandas as pd
import itertools

class Tree(object):
    """
    A class to represent a phylogenetic tree using ete3 and a dictionary for efficient traversal.
    """
    def __init__(self, tree_file=None):
        self.phylotree = None
        self.root = None
        if tree_file is not None:
            self.load_newick(tree_file)

    def simulate_tree(self, n_leaves, n_internal_nodes, n_branches, branch_length_mean, branch_length_std, trait_mean, trait_std):
        self.phylotree = PhyloTree.simulate_tree(n_leaves, n_internal_nodes, n_branches, branch_length_mean, branch_length_std)
        self.root = self.phylotree.get_tree_root()

    def get_species_cov_matrix(self):
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
        return species_cov_matrix


    def load_newick(self, tree_file):
        self.phylotree = PhyloTree(tree_file)
        self.root = self.phylotree.get_tree_root()
    
    def save_newick(self, tree_file):
        self.phylotree.write(format=1, outfile=tree_file)