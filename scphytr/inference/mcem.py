import numpy as np

import jax.numpy as jnp
import jax
import optax
from functools import partial
from tqdm import tqdm

from .base import Base

class MCEM(Base):
    """
    Monte Carlo Expectation-Maximization for estimating the parameters of a trait model with missing data. 
    The E-step uses importance sampling to estimate the missing data.
    The M-step is a simple gradient descent step. The two steps are alternated until convergence.
    """
    def __init__(self, tree, trait_model, lr=0.1, **kwargs):
        super().__init__(tree, trait_model, **kwargs)
        # The nodes contain a dictionary of values for each trait
        self.lr = lr
        self.trace = []
        self.opt = optax.adam(self.lr)


    @partial(jax.jit, static_argnums=0)  # treat `self` as static; make sure it’s pytree-safe
    def step(self, params, opt_state, observed_trait_values, rng, n_samples=100):
        def loss_fn(rng, p):
            trait_params = p[:len(self.trait_model.sample_parameters())]
            observed_params = p[len(self.trait_model.sample_parameters()):]
            # Sample from the trait model prior with current parameters
            trait_sample = self.trait_model.sample_prior(trait_params, rng)
            return -self.observation_model.logpdf(observed_trait_values, observed_params, trait_sample)

        def batch_loss_fn(rng):
            # Average over a batch of Monte Carlo samples
            rngs = jax.random.split(rng, n_samples)
            vectorized_loss_fn = jax.vmap(loss_fn, in_axes=(0, None))
            # Compute Monte Carlo estimate of the loss
            return jnp.mean(vectorized_loss_fn(rngs))

        loss, grads = jax.value_and_grad(batch_loss_fn)(rng)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def run_em(self, n_steps=100, seed=42, n_samples=100):
        # Initialize the parameters of the model
        trait_params = [jnp.array(p) for p in self.trait_model.sample_parameters()]
        observed_params = [jnp.array(p) for p in self.observation_model.sample_parameters()]
        params = trait_params + observed_params

        opt_state = self.opt.init(params)

        observed_trait_values = self.trait_model.reshape_trait_values(self.tree.get_trait_values())

        rng = jax.random.PRNGKey(seed)
        # A simple update loop.
        self.trace = []
        pbar = tqdm(range(n_steps), desc=f"MCEM on {self.trait_model.__class__.__name__}")
        for _ in pbar:
            rng, rng_new = jax.random.split(rng)
            params, opt_state, loss = self.step(params, opt_state, observed_trait_values, rng_new, n_samples)
            self.trace.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        return params

    def fit_trait_model(self, **kwargs):
        params = self.run_em(**kwargs)
        self.trait_model.set_parameters(params)
