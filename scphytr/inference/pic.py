import numpy as np

from .base import Base

class PIC(Base):
    def __init__(self, tree, trait_model, observation_model, method_kwargs):
        super().__init__(tree, trait_model, observation_model, method_kwargs)
        # The nodes contain a dictionary of values for each trait

    def estimate_trait_evolutionary_rates(self): # should be able to do this for multiple traits at once, vectorized
        # tree must contain branch lengths for all nodes except root, and trait values for all leaves
        root = self.tree.root
        trait_names = self.tree.get_trait_names()

        standardized_contrasts = [np.zeros(len(trait_names))] # n-1 contrasts for each trait, n is the number of species
        
        def new_trait(t1, t2, v1, v2):
            return ((1/v1)*t1 + (1/v2)*t2)/(1/v1 + 1/v2)
        
        def new_length(vk, v1, v2):
            return vk + v1*v2/(v1+v2)

        # Pruning algorithm
        def descend(root):
            trait_values = [np.zeros(len(trait_names))] 
            branch_lengths = []
            for child in root.get_children():
                child_trait_value, child_branch_length = descend(child)
                trait_values.append(child_trait_value)
                branch_lengths.append(child_branch_length)
            
            if len(trait_values) > 0:
                new_trait_value = new_trait(trait_values[0], trait_values[1], branch_lengths[0], branch_lengths[1])
                new_branch_length = new_length(root.dist, branch_lengths[0], branch_lengths[1])
                standardized_contrast = (trait_values[-1]-trait_values[0])/sum(branch_lengths)
                standardized_contrasts.append(standardized_contrast)
            else:
                new_trait_value = np.array(root.trait[trait_names])
                new_branch_length = root.dist

            return new_trait_value, new_branch_length
        
        descend(root)

        est_rates = np.sum(np.array(standardized_contrasts)**2, axis=0)/len(standardized_contrasts)
        return est_rates, standardized_contrasts

    def fit_trait_model(self):
        est_rates, _ = self.estimate_trait_evolutionary_rates()
        self.trait_model.set_trait_cov_matrix(est_rates)
        return est_rates