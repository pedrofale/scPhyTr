import numpy as np

import jax.numpy as jnp
import jax
import optax
from functools import partial
from tqdm import tqdm

from .base import BaseInference

class ML(BaseInference):
    """
    Maximum likelihood estimation for estimating the parameters of a trait model.
    TODO: allow minibatching. Tricky because the observations are not independent -- there's a tree!
    """
    def __init__(self, tree, trait_model, lr=0.1, **kwargs):
        super().__init__(tree, trait_model, **kwargs)
        # The nodes contain a dictionary of values for each trait
        self.lr = lr
        self.trace = []
        self.opt = optax.adam(self.lr)

    @partial(jax.jit, static_argnums=0)  # treat `self` as static; make sure it’s pytree-safe
    def step(self, params, opt_state, trait_values):
        def loss_fn(p):
            return -self.trait_model.logpdf_prior(trait_values, p)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def run_ml(self, n_steps=100):
        # Initialize the parameters of the model    
        params = [jnp.array(p) for p in self.trait_model.sample_parameters()]

        opt_state = self.opt.init(params)

        trait_values = self.trait_model.reshape_trait_values(self.tree.get_trait_values())

        # A simple update loop.
        self.trace = []
        pbar = tqdm(range(n_steps), desc=f"SGD on {self.trait_model.__class__.__name__}")
        for _ in pbar:
            params, opt_state, loss = self.step(params, opt_state, trait_values)
            self.trace.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        return params

    def fit_trait_model(self, **kwargs):
        params = self.run_ml(**kwargs)
        self.trait_model.set_parameters(params)
