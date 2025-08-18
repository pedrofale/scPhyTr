from scphytr.inference import PIC

def estimate_evolutionary_correlation(adata, characters): # Populates the adata.uns with a correlation matrix for the specified characters
    tree = adata.uns['tree']

    # Store the traits in the leaves
    species_trait_values = make_trait_values(adata, 'species', characters)
    tree.set_trait_values(species_trait_values)

    # Create trait and observation models
    trait_model = trait_models[trait_model](tree, learnable_parameters=['rates']) # Learn the evolutionary rates of each trait independently
    observation_model = observation_models[observation_model](tree)

    if method == 'mcmc':
        alg = MCMC(tree, trait_model, observation_model, method_kwargs=method_kwargs)
        estimated_rates = alg.estimate_global_rate()
    elif method == 'vi':
        alg = BBVI(tree, trait_model, observation_model, method_kwargs=method_kwargs)
        estimated_rates = alg.estimate_global_rate()
    elif method == 'pic':
        if trait_model != 'brownian_motion':
            raise ValueError("PIC is only available for Brownian motion")
        alg = PIC(tree, trait_model, observation_model=observation_model, method_kwargs=method_kwargs)
        # fit the trait model
        alg.fit_trait_model()
        # test for trait correlations using the contrats
        estimated_rates = alg.estimate_correlations()
    elif method == 'plgs':
        pass
    else:
        raise ValueError(f"Method {method} not supported")
    
    adata.uns['estimated_rates'] = estimated_rates
    return alg
