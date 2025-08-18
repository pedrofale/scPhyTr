from scphytr.inference import MCMC, BBVI, PIC, SGD
from scphytr.trait_models import trait_models
from scphytr.observation_models import observation_models

import numpy as np

def make_trait_values(adata, species_obs, characters):
    trait_values = dict()
    for species in adata.obs[species_obs].unique():
        trait_values[species] = {character: np.mean(adata[adata.obs[species_obs] == species][character]) for character in characters}
    return trait_values


def fit_model(adata, characters, trait_model, observation_model, method='mcmc', method_kwargs={}): # Populates the adata.uns with a global rate for the specified characters using the specified model
    tree = adata.uns['tree']

    # Store the traits in the leaves
    species_trait_values = make_trait_values(adata, 'species', characters)
    tree.set_trait_values(species_trait_values)

    # Create trait and observation models
    trait_model = trait_models[trait_model](tree, learnable_parameters=['rates']) # Learn the evolutionary rates of each trait independently
    observation_model = observation_models[observation_model](tree)

    if method == 'mcmc':
        alg = MCMC(tree, trait_model, observation_model, method_kwargs=method_kwargs)
    elif method == 'vi':
        alg = BBVI(tree, trait_model, observation_model, method_kwargs=method_kwargs)
    elif method == 'sgd':
        alg = SGD(tree, trait_model, observation_model, method_kwargs=method_kwargs)
    elif method == 'pic':
        if trait_model != 'brownian_motion':
            raise ValueError("PIC is only available for Brownian motion")
        alg = PIC(tree, trait_model, observation_model=observation_model, method_kwargs=method_kwargs)
    elif method == 'plgs':
        pass
    else:
        raise ValueError(f"Method {method} not supported")

    alg.fit()

    # Update anndata with the fitted model
    for parameter in trait_model.get_learnable_parameters():
        adata.uns[f'{parameter}'] = alg.trait_model.get_parameter_estimate(parameter)
    
    # Update anndata.obs with per-cell parameters 
    # Update anndata.var with per-gene parameters

    return alg
