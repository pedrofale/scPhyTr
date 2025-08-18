import jax.numpy as jnp
import jax
import optax
import functools

from .base import BaseInference

class BBVI(BaseInference):
    """
    Black box variational inference for estimating the parameters of a trait model.
    """
    def __init__(self, tree, lr=0.1, optimizer='adam', **kwargs):
        super().__init__(tree, **kwargs)
        # The nodes contain a dictionary of values for each trait
        self.lr = lr
        self.optimizer = optimizer


    def elbo(self, x, params, variational_parameters):        
        trait_variational_parameters = variational_parameters[0]
        observation_variational_parameters = variational_parameters[1]

        # Sample from the trait model
        trait_sample = self.trait_variational_distribution.sample(trait_variational_parameters)
        trait_kl = self.trait_model.kl_divergence(trait_variational_parameters)

        # Sample from the observation model
        observation_sample = self.observation_variational_distribution.sample(observation_variational_parameters)
        observation_kl = self.observation_model.kl_divergence(observation_variational_parameters)

        return jnp.mean(self.observation_model.log_likelihood(x, params, observation_sample, trait_sample)) + trait_kl + observation_kl


    def run_vi(self, n_steps=100):
        # Initialize the parameters of the trait model
        trait_params = jnp.array(self.trait_model.sample_parameters())
        trait_variational_parameters = jnp.array(self.trait_model.sample_parameters())
        observation_params = jnp.array(self.trait_model.sample_parameters())
        observation_variational_parameters = jnp.array(self.trait_model.sample_parameters())
        
        trait_variational_optimizer = optax.adam(self.lr)
        trait_variational_opt_state = trait_variational_optimizer.init(trait_variational_parameters)
        observation_variational_optimizer = optax.adam(self.lr)
        observation_variational_opt_state = observation_variational_optimizer.init(observation_variational_parameters)
        trait_params_optimizer = optax.adam(self.lr)
        trait_params_opt_state = trait_params_optimizer.init(trait_params)
        observation_params_optimizer = optax.adam(self.lr)
        observation_params_opt_state = observation_params_optimizer.init(observation_params)                

        # A simple update loop.
        for _ in range(n_steps):
            grads = jax.grad(self.elbo)(params, variational_parameters)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        return params

    def fit_trait_model(self):
        params = self.run_vi()
        self.trait_model.set_parameters(params)
