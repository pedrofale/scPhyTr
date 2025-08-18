import numpy as np

import jax.numpy as jnp
import jax
import optax
from functools import partial
from tqdm import tqdm

from src.scphytr import observation_models

from .base import BaseInference

class VBEM(BaseInference):
    """
    Variational Expectation-Maximization for estimating the parameters of a trait model with missing data. 
    The E-step uses variational inference to estimate the missing data.
    The M-step is a simple gradient descent step. The two steps are alternated until convergence.
    """
    def __init__(self, tree, trait_model, lr=0.1, **kwargs):
        super().__init__(tree, trait_model, **kwargs)
        # The nodes contain a dictionary of values for each trait
        self.lr = lr
        self.trace = []
        self.parameter_optimizer = optax.adam(self.lr)
        self.variational_optimizer = optax.adam(self.lr)

    @partial(jax.jit, static_argnums=0)  # treat `self` as static; make sure it’s pytree-safe
    def m_step(self, params, opt_state, observed_trait_values, trait_samples, weights):
        def loss_fn(trait_sample, weight, p): # single-sample loss function
            trait_params = p[:len(self.trait_model.sample_parameters())]
            observed_params = p[len(self.trait_model.sample_parameters()):]
            lp = self.trait_model.logpdf_prior(trait_sample, trait_params)
            ll = self.observation_model.logpdf(observed_trait_values, trait_sample, [jnp.log(self.observation_model.std)])
            ml = (lp + ll)*weight    
            return -ml

        def batch_loss_fn(params, samples, weights):
            # Average over a batch of Monte Carlo samples
            vectorized_loss_fn = jax.vmap(loss_fn, in_axes=(0, 0, None))
            # Compute Importance-weighted Monte Carlo estimate of the loss
            return jnp.sum(vectorized_loss_fn(samples, weights, params))

        loss, grads = jax.value_and_grad(batch_loss_fn)(params, trait_samples, weights)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jax.jit, static_argnums=(0, 1))  # treat `self` as static; make sure it’s pytree-safe
    def sample_and_compute_weights(self, n_samples, params, rng, observed_trait_values):
        def weights_fn(rng, p): # single-sample loss function
            trait_params = p[:len(self.trait_model.sample_parameters())]
            observed_params = p[len(self.trait_model.sample_parameters()):]
            trait_sample = self.trait_model.sample_proposal(trait_params, rng)
            lq = self.trait_model.logpdf_proposal(trait_sample, trait_params)
            lp = self.trait_model.logpdf_prior(trait_sample, trait_params)
            ll = self.observation_model.logpdf(observed_trait_values, trait_sample, [jnp.log(self.observation_model.std)])
            lw = ll + lp - lq
            return lw, trait_sample

        def batch_weights_fn(rng, params):
            # Normalize over a batch of Monte Carlo samples
            rngs = jax.random.split(rng, n_samples)
            vectorized_weights_fn = jax.vmap(weights_fn, in_axes=(0, None))
            lw, trait_samples = vectorized_weights_fn(rngs, params)
            logZ = jax.scipy.special.logsumexp(lw)
            weights = jnp.exp(lw - logZ)
            return weights, trait_samples

        return batch_weights_fn(rng, params) # returns weights and trait samples

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
        self.esss = []
        pbar = tqdm(range(n_steps), desc=f"MCEM on {self.trait_model.__class__.__name__}")
        weights, proposal_samples = self.sample_and_compute_weights(n_samples, params, rng, observed_trait_values)
        for _ in pbar:
            rng, rng_new = jax.random.split(rng)
            # Sample from proposal distribution
            weights, proposal_samples = self.sample_and_compute_weights(n_samples, params, rng_new, observed_trait_values)
            # print(weights)
            # Take M-step with importance weights
            opt_state = self.opt.init(params)
            print('M-step')
            for i in range(5):
                params, opt_state, loss = self.m_step(params, opt_state, observed_trait_values, proposal_samples, weights)
                print(loss)
            ess = jnp.sum(weights)**2 / jnp.sum(weights**2)
            self.trace.append(loss)
            self.esss.append(ess)
            pbar.set_postfix({"loss": f"{loss:.4f}", "ESS": f"{ess:.4f}/{n_samples}"})

        return params

    def fit_trait_model(self, **kwargs):
        params = self.run_em(**kwargs)
        self.trait_model.set_parameters(params[:len(self.trait_model.sample_parameters())])
        self.observation_model.set_parameters(params[len(self.trait_model.sample_parameters()):])
