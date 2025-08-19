import jax.numpy as jnp
import jax
import optax
from functools import partial
from tqdm import tqdm

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
    def m_step(self, trait_samples, params, opt_state, variational_params, observed_trait_values):
        def elbo_fn(trait_sample, p, vp): # single-sample loss function
            trait_params = p[:len(self.trait_model.sample_parameters())]
            observed_params = p[len(self.trait_model.sample_parameters()):]
            trait_kl = self.trait_model.kl_divergence(trait_params, vp)
            ll = self.observation_model.logpdf(observed_trait_values, trait_sample, observed_params)
            elbo = ll - trait_kl
            return elbo

        def batch_elbo_fn(params, samples):
            # Average over a batch of Monte Carlo samples
            vectorized_elbo_fn = jax.vmap(elbo_fn, in_axes=(0, None, None))
            elbo = jnp.mean(vectorized_elbo_fn(samples, params, variational_params))
            return -elbo

        loss, grads = jax.value_and_grad(batch_elbo_fn)(params, trait_samples)
        updates, opt_state = self.parameter_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jax.jit, static_argnums=(0, 1))  # treat `self` as static; make sure it’s pytree-safe
    def e_step(self, n_samples, variational_params, variational_opt_state, params, observed_trait_values, rng):
        def elbo_fn(rng, p, vp): # single-sample loss function
            trait_params = p[:len(self.trait_model.sample_parameters())]
            observed_params = p[len(self.trait_model.sample_parameters()):]
            trait_sample = self.trait_model.sample_variational_proposal(vp, rng)
            trait_kl = self.trait_model.kl_divergence(trait_params, vp)
            ll = self.observation_model.logpdf(observed_trait_values, trait_sample, observed_params)
            elbo = ll - trait_kl
            return elbo

        def batch_elbo_fn(variational_params, params, rng):
            # Average over a batch of Monte Carlo samples
            rngs = jax.random.split(rng, n_samples)
            vectorized_elbo_fn = jax.vmap(elbo_fn, in_axes=(0, None, None))
            elbo = jnp.mean(vectorized_elbo_fn(rngs, params, variational_params))
            return -elbo

        loss, grads = jax.value_and_grad(batch_elbo_fn)(variational_params, params, rng)
        updates, variational_opt_state = self.variational_optimizer.update(grads, variational_opt_state, variational_params)
        variational_params = optax.apply_updates(variational_params, updates)
        return variational_params, variational_opt_state, loss

    def run_em(self, n_steps=100, seed=42, n_samples=10, n_inner_steps=5):
        # Initialize the parameters of the model
        trait_params = [jnp.array(p) for p in self.trait_model.sample_parameters()]
        observed_params = [jnp.array(p) for p in self.observation_model.sample_parameters()]
        params = trait_params + observed_params

        # Initialize the variational parameters
        variational_params = [jnp.array(p) for p in self.trait_model.sample_variational_parameters()]

        opt_state = self.parameter_optimizer.init(params)
        variational_opt_state = self.variational_optimizer.init(variational_params)

        observed_trait_values = self.trait_model.reshape_trait_values(self.tree.get_trait_values())

        rng = jax.random.PRNGKey(seed)
        # A simple update loop.
        self.trace = []
        pbar = tqdm(range(n_steps), desc=f"VBEM on {self.trait_model.__class__.__name__}")
        for _ in pbar:
            rng, rng_new = jax.random.split(rng)
            # Fit q(z)
            # print("E-step")
            variational_opt_state = self.variational_optimizer.init(variational_params)
            for _ in range(n_inner_steps):
                variational_params, variational_opt_state, variational_loss = self.e_step(n_samples, variational_params, variational_opt_state, params, observed_trait_values, rng_new)
                # print(variational_loss)
            # Take one sample for the M-step
            rngs = jax.random.split(rng_new, n_samples)
            variational_samples = jax.vmap(self.trait_model.sample_variational_proposal, in_axes=(None, 0))(variational_params, rngs)
            # Update parameters
            # print("M-step")
            opt_state = self.parameter_optimizer.init(params)
            for _ in range(n_inner_steps):
                params, opt_state, loss = self.m_step(variational_samples, params, opt_state, variational_params, observed_trait_values)
                self.trace.append(loss)
                # print(loss)
            self.trace.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        return params, variational_params

    def fit_trait_model(self, **kwargs):
        params, variational_params = self.run_em(**kwargs)
        self.trait_model.set_parameters(params[:len(self.trait_model.sample_parameters())])
        self.observation_model.set_parameters(params[len(self.trait_model.sample_parameters()):])
        self.trait_model.set_variational_parameters(variational_params)
