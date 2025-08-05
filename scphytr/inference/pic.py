import numpy as np
import statsmodels.api as sm

from .base import Base

class PIC(Base):
    """
    Phylogenetic independent contrats (PIC) for estimating the evolutionary rates of traits under a Brownian motion model.
    """
    def __init__(self, tree, trait_model, observation_model=None, specific_param=None):
        super().__init__(tree, trait_model, observation_model=observation_model)
        # The nodes contain a dictionary of values for each trait
        self.specific_param = specific_param
        self.standardized_contrasts = None

    def run_pic(self): # should be able to do this for multiple traits at once, vectorized
        # tree must contain branch lengths for all nodes except root, and trait values for all leaves
        root = self.tree.root
        standardized_contrasts = [] # n-1 contrasts for each trait, n is the number of species
        
        def new_trait(t1, t2, v1, v2):
            return ((1/v1)*t1 + (1/v2)*t2)/(1/v1 + 1/v2)

        def new_length(vk, v1, v2):
            return vk + (v1*v2)/(v1+v2)

        # Pruning algorithm
        def descend(root):
            trait_values = [] 
            branch_lengths = []
            for child in root.get_children():
                child_trait_value, child_branch_length = descend(child)
                trait_values.append(child_trait_value)
                branch_lengths.append(child_branch_length)
            
            if len(trait_values) > 1:
                standardized_contrast = (trait_values[0]-trait_values[1])/sum(branch_lengths)
                standardized_contrasts.append(standardized_contrast)        
                
                new_trait_value = new_trait(trait_values[0], trait_values[1], branch_lengths[0], branch_lengths[1])
                new_branch_length = new_length(root.dist, branch_lengths[0], branch_lengths[1])
            else:
                new_trait_value = np.array([root.trait[n] for n in root.trait])# - self.trait_model.trait_means.values.ravel()
                new_branch_length = root.dist

            return new_trait_value, new_branch_length

        root_trait_value, _ = descend(root)

        est_rates = np.sum(np.array(standardized_contrasts)**2, axis=0)/len(standardized_contrasts)
        return est_rates, root_trait_value, standardized_contrasts

    def test_trait_evolutionary_correlations(self):
        # test for trait correlations using the standardized contrasts
        trait_correlations = np.zeros((len(self.standardized_contrasts), len(self.standardized_contrasts)))
        for trait_1 in range(len(self.standardized_contrasts)):
            for trait_2 in range(trait_1+1, len(self.standardized_contrasts)):
                # Fit regression line to the standardized contrasts enforcing origin
                res = sm.linregress(self.standardized_contrasts[trait_1], self.standardized_contrasts[trait_2]).fit()
                trait_correlations[trait_1, trait_2] = res.pvalues[0]
                trait_correlations[trait_2, trait_1] = res.pvalues[0]
        return trait_correlations

    def fit_trait_model(self):
        est_rates, est_root_values, standardized_contrasts = self.run_pic()
        self.standardized_contrasts = standardized_contrasts
        self.trait_model.set_trait_cov_matrix(est_rates) # assuming each trait follows an independent Brownian motion model. Set the trait covariance matrix the diagonal matrix containing the estimated rates
        self.trait_model.set_trait_means(est_root_values)
