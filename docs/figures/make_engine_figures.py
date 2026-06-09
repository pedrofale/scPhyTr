"""Figure for docs/inference_engines.md: three E-step engines on one posterior.

At a fixed eta on the Poisson/BM running example we compare, for one latent
node, the posterior marginal as approximated by (i) the Laplace Gaussian,
(ii) importance sampling with that Gaussian as proposal, and (iii) NUTS. We also
show the self-normalized importance-weight distribution and its ESS.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

import jax
import blackjax

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance
from scphytr.inference.laplace import PoissonObservation
from scphytr.inference.tree_laplace_mv import _MVTreeModel, _newton_mode
from scphytr.inference.estep import _prior_quad_batch, MCMCEStep

HERE = os.path.dirname(__file__)


def balanced_newick(depth, root_branch=0.5):
    c = [0]

    def rec(d, root):
        if d == 0:
            c[0] += 1
            return f"L{c[0]}:1.0"
        s = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return s if root else s + ":1.0"

    return rec(depth, True) + f":{root_branch};"


def corr_cov(rho, sds):
    R = np.array([[1.0, rho], [rho, 1.0]])
    D = np.diag(sds)
    return D @ R @ D


def main():
    rng = np.random.default_rng(0)
    tree = Tree(balanced_newick(5, root_branch=0.5))      # 32 leaves
    n = len(tree.root.get_leaves())
    K = corr_cov(0.8, [0.9, 1.1]); mu = np.array([2.0, 1.6])
    C = bm_covariance(tree)
    Z = rng.multivariate_normal(np.tile(mu, n), np.kron(C, K)).reshape(n, 2)
    S = rng.uniform(40, 80, size=n)
    Y = rng.poisson(S[:, None] * np.exp(Z))
    obs = PoissonObservation(Y, S)

    M = _MVTreeModel(tree, 0.0, mu, K)
    p = M.p
    mode = _newton_mode(M, obs)
    leaf = M.leaf_node_idx
    Wdiag = np.zeros((M.N, p)); Wdiag[leaf] = obs.neg_hess_diag(mode[leaf])
    Sig_an, _ = M.posterior_covariances(Wdiag)

    node = M.root_idx; g = 0                              # root, gene 0 (broad posterior)
    lap_mean = mode[node, g]; lap_sd = np.sqrt(Sig_an[node, g, g])

    # ---- importance sampling with the Laplace Gaussian proposal ----
    Zs = M.sample_gaussian(Wdiag, mode, 40000, np.random.default_rng(1))
    delta = Zs - mode[None]
    prior_q = _prior_quad_batch(M, Zs, centered=False)
    prop_q = _prior_quad_batch(M, delta, centered=True) + np.einsum("snp,np->s", delta ** 2, Wdiag)
    loglik = np.array([obs.loglik(Zs[s, leaf]) for s in range(Zs.shape[0])])
    logw = loglik - 0.5 * prior_q + 0.5 * prop_q
    wn = np.exp(logw - logsumexp(logw))
    ess = 1.0 / np.sum(wn ** 2)
    is_samples = Zs[:, node, g]

    # ---- NUTS ----
    eng = MCMCEStep(n_samples=4000, n_warmup=1000, seed=0)
    logdensity, N, _ = eng._logdensity_fn(M, obs)
    import jax.numpy as jnp
    z0 = jnp.asarray(mode.reshape(-1))
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, params), _ = warmup.run(k1, z0, num_steps=1000)
    kernel = blackjax.nuts(logdensity, **params)

    def step(carry, key):
        st, info = kernel.step(key, carry)
        return st, st.position

    _, pos = jax.lax.scan(step, state, jax.random.split(k2, 4000))
    mcmc_samples = np.asarray(pos).reshape(4000, N, p)[:, node, g]

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = np.linspace(lap_mean - 4 * lap_sd, lap_mean + 4 * lap_sd, 300)
    lap_pdf = np.exp(-0.5 * ((xs - lap_mean) / lap_sd) ** 2) / (lap_sd * np.sqrt(2 * np.pi))
    ax[0].plot(xs, lap_pdf, "k-", lw=2, label="Laplace Gaussian")
    ax[0].hist(is_samples, bins=60, weights=wn, density=True, alpha=0.5,
               color="#2c7fb8", label="importance sampling")
    ax[0].hist(mcmc_samples, bins=50, density=True, alpha=0.5,
               color="#d95f0e", label="NUTS")
    ax[0].set_title("Posterior marginal of one latent node\n(root, gene 0; fixed eta)", fontsize=11)
    ax[0].set_xlabel("$z$"); ax[0].set_ylabel("posterior density"); ax[0].legend(fontsize=9)

    order = np.sort(wn)[::-1]
    ax[1].plot(np.arange(1, len(order) + 1), order, color="#2c7fb8")
    ax[1].set_yscale("log"); ax[1].set_xscale("log")
    ax[1].set_title(f"Self-normalized importance weights\nESS = {ess:.0f} / {len(wn)} "
                    f"({100*ess/len(wn):.0f}%)", fontsize=11)
    ax[1].set_xlabel("sample rank"); ax[1].set_ylabel("normalized weight $\\tilde w_s$")

    fig.tight_layout()
    out = os.path.join(HERE, "engines_posterior.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")
    print(f"node=root gene0: Laplace mean={lap_mean:.3f} sd={lap_sd:.3f}")
    print(f"IS:   mean={np.sum(wn*is_samples):.3f} sd={np.sqrt(np.sum(wn*(is_samples-np.sum(wn*is_samples))**2)):.3f} ess={ess:.0f}")
    print(f"NUTS: mean={mcmc_samples.mean():.3f} sd={mcmc_samples.std():.3f}")


if __name__ == "__main__":
    main()
