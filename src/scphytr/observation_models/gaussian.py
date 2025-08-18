import numpy as np
import jax.scipy as jsp
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import jax
from .base import BaseObservationModel

class Gaussian(BaseObservationModel):
    def __init__(self, std, learnable_parameters=['std']):
        self.learnable_parameters = learnable_parameters
        self.std = std

    def simulate_observations(self, trait_values, seed=42):
        rng = jax.random.PRNGKey(seed)
        return tfd.Normal(loc=trait_values, scale=self.std).sample(rng)

    def logpdf(self, observations, trait_values, params):
        log_std = params[0]
        return tfd.Normal(loc=trait_values, scale=jnp.exp(log_std)).log_prob(observations).sum()

    def sample_parameters(self):
        return [jnp.log(self.std)]

    def set_parameters(self, params):
        self.std = jnp.exp(params[0])