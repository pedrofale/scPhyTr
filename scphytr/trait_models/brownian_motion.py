import numpy as np
import scphytr.utils as utils

def fit_pic(root):
    # tree must contain branch lengths for all nodes except root, and trait values for all leaves
    standardized_contrasts = []
    
    def new_trait(t1, t2, v1, v2):
        return ((1/v1)*t1 + (1/v2)*t2)/(1/v1 + 1/v2)
    
    def new_length(vk, v1, v2):
        return vk + v1*v2/(v1+v2)

    # Pruning algorithm
    def descend(root):
        trait_values = []
        branch_lengths = []
        for child in root['children']:
            child_trait_value, child_branch_length = descend(child)
            trait_values.append(child_trait_value)
            branch_lengths.append(child_branch_length)
        
        new_trait_value = root['trait']
        new_branch_length = root['branch_length']
        if len(trait_values) > 0:
            new_trait_value = new_trait(trait_values[0], trait_values[1], branch_lengths[0], branch_lengths[1])
            new_branch_length = new_length(root['branch_length'], branch_lengths[0], branch_lengths[1])
            standardized_contrast = (trait_values[-1]-trait_values[0])/sum(branch_lengths)
            standardized_contrasts.append(standardized_contrast)

        return new_trait_value, new_branch_length
    
    descend(root)

    est_rate = sum(np.array(standardized_contrasts)**2)/len(standardized_contrasts)
    return est_rate, standardized_contrasts
    

def fit_ml(root):
    

def fit_mcmc(root, trait):

def fit_vi(root, trait):


class BrownianMotion(object):
    def __init__(self, tree, trait_means, trait_cov_matrix):
        self.tree = tree # ete3 tree
        self.n_species = len(self.tree.get_species())
        self.trait_means = trait_means # means of traits
        self.trait_cov_matrix = trait_cov_matrix # correlations between traits -- TODO: extend to per-clade trait covariance matrices
        self.species_cov_matrix = utils.tree.make_species_cov_matrix(tree)

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
        dW = np.random.multivariate_normal(0, cov_matrix*dt, N)  # Increments for every trait
        W = np.concatenate(([0], np.cumsum(dW)))  # Cumulative sum to get the path
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
        species_paths = []
        def descend(root, path):
            if root.is_leaf():
                species_paths.append(path)
            for child in root.children:
                local_path = self.multivariate_brownian_motion_path(child['branch_length'], N, self.trait_cov_matrix)
                path = np.concatenate((path, local_path))
                descend(child, path)
        
        path = self.trait_means + self.multivariate_brownian_motion_path(self.tree.root['branch_length'], N, self.trait_cov_matrix)
        return descend(self.tree.root, path)

    def simulate_traits(self, seed=42):
        np.random.seed(seed)
        # Create variance-covariance matrix
        a = np.repeat(self.trait_means, self.n_species) 
        V = np.outer(self.trait_cov_matrix, self.species_cov_matrix) 
        species_trait_values = np.random.multivariate_normal(a, V) 
        species_trait_values = species_trait_values.reshape(self.n_species, -1)
        return species_trait_values

