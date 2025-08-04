from scphytr.inference import MCMC, BBVI, PIC
from scphytr.trait_models import trait_models
from scphytr.observation_models import observation_models

import numpy as np

def estimate_global_rate(adata, characters, trait_model, observation_model, method='mcmc', method_kwargs={}): # Populates the adata.uns with a global rate for the specified characters using the specified model
    tree = adata.uns['tree']

    # Store the traits in the tree nodes
    for node in tree.phylotree.traverse():
        node.trait = {character: np.mean(adata[adata.obs['species'] == node.name][character]) for character in characters}

    # Create trait and observation models
    trait_model = trait_models[trait_model](tree, learnable_parameters=['rates']) # Learn the evolutionary rates of each trait independently
    observation_model = observation_models[observation_model](tree)

    if method == 'mcmc':
        alg = MCMC(tree, trait_model, observation_model, method_kwargs)
        estimated_rates = alg.estimate_global_rate()
    elif method == 'vi':
        alg = BBVI(tree, trait_model, observation_model, method_kwargs)
        estimated_rates = alg.estimate_global_rate()
    elif method == 'pic':
        if trait_model != 'brownian_motion':
            raise ValueError("PIC is only available for Brownian motion")
        alg = PIC(tree, trait_model, observation_model, method_kwargs)
        estimated_rates = alg.estimate_global_rate()
    else:
        raise ValueError(f"Method {method} not supported")
    
    adata.uns['estimated_rates'] = estimated_rates
    return alg

def estimate_lineage_rates(adata, lineage): # Populates the adata.uns with a rate for each lineage
    pass

def estimate_state_rates(adata, state): # Populates the adata.uns with a rate for each state
    pass