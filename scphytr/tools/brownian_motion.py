def fit_pic(tree):
    # tree must contain branch lengths for all nodes except root, and trait values for all leaves
    standardized_contrasts = []
    
    def new_trait(t1, t2, v1, v2):
        return ((1/v1)*t1 + (1/v2)*t2)/(1/v1 + 1/v2)
    
    def new_length(vk, v1, v2):
        return vk + v1*v2/(v1+v2)

    # Pruning algorithm
    def descend(node):
        trait_values = []
        branch_lengths = []
        for child in node['children']:
            child_trait_value, child_branch_length = descend(child)
            trait_value += child_trait_value
            trait_values.append(child_trait_value)
            branch_lengths.append(child_branch_length)
        
        new_trait_value = node['trait']
        new_branch_length = node['branch_length']
        if len(trait_values) > 0:
            new_trait_value = new_trait(trait_values[0], trait_values[1], branch_lengths[0], branch_lengths[1])
            new_branch_length = new_length(node['branch_length'], branch_lengths[0], branch_lengths[1])
            standardized_contrast = (trait_values[-1]-trait_values[0])/sum(branch_lengths)
            standardized_contrasts.append(standardized_contrast)

        return new_trait_value, new_branch_length
    
    descend(tree)

    est_rate = sum(standardized_contrasts**2)/len(standardized_contrasts)
    return est_rate, standardized_contrasts
    

def fit_ml(tree):
    

def fit_mcmc(tree, trait):

def fit_vi(tree, trait):

class BrownianMotion(object):
    def __init__(self, tree):
        pass

    def simulate(self):
        pass

    def fit(self):
        pass

    def 