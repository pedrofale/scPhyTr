def estimate_global_rate(adata, character, trait_model, observation_model, method='mcmc', method_kwargs={}): # Populates the adata.uns with a global rate for the specified character using the specified model
    tree = adata.uns['tree']
    trait_values = adata.obs[character]
    trait_model.fit(tree, trait_values)
    observation_model.fit(tree, trait_values)
    if method == 'mcmc':
        inference.mcmc(trait_model, observation_model, method_kwargs)
    pass

def estimate_lineage_rates(adata, lineage): # Populates the adata.uns with a rate for each lineage
    pass

def estimate_state_rates(adata, state): # Populates the adata.uns with a rate for each state
    pass