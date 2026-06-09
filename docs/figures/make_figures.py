"""Generate the figures and the running-example numbers used in docs/methods.md.

Outputs (into docs/figures/):
  * precision_vs_covariance.png  -- sparse tree precision Q vs dense covariance Q^{-1}
  * running_example.png          -- the simulation woven through the write-up

Run inside the `scphytr` conda env. Deterministic (seed 0).
"""

import os
import io
import time
import contextlib

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance
from scphytr.inference.laplace import PoissonObservation, GaussianObservation
from scphytr.tools.estimation import fit_bm_mv, fit_mv_latent
from scphytr.tools.em import fit_mv_em

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


# --------------------------------------------------------------------------
# Figure 1: why the precision is sparse (joint over ALL nodes), free root.
# --------------------------------------------------------------------------
def joint_bm_precision(tree, root_var):
    """Dense precision Q of a unit-rate BM joint over all nodes (free root).

    Edge u->pa with branch variance v_u contributes 1/v_u to the (u,u) and
    (pa,pa) diagonals and -1/v_u to the (u,pa) off-diagonals. The root gets a
    prior variance ``root_var`` (a 'dummy parent'), making Q SPD.
    Nodes are ordered breadth-first so parent-child couplings are visible.
    """
    nodes = list(tree.root.traverse("levelorder"))
    idx = {nd: i for i, nd in enumerate(nodes)}
    N = len(nodes)
    Q = np.zeros((N, N))
    for nd in nodes:
        i = idx[nd]
        if nd is tree.root:
            Q[i, i] += 1.0 / root_var
            continue
        v = nd.dist
        pa = idx[nd.up]
        Q[i, i] += 1.0 / v
        Q[pa, pa] += 1.0 / v
        Q[i, pa] -= 1.0 / v
        Q[pa, i] -= 1.0 / v
    return Q, nodes


def figure_precision_vs_covariance():
    tree = Tree(balanced_newick(3, root_branch=0.5))     # 8 leaves, 15 nodes
    Q, nodes = joint_bm_precision(tree, root_var=0.5)
    Sigma = np.linalg.inv(Q)
    N = Q.shape[0]
    density = (np.abs(Q) > 1e-12).sum() / Q.size

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax[0].imshow(np.abs(Q) > 1e-12, cmap="Greys", interpolation="none")
    ax[0].set_title(f"Prior precision $Q$ (joint over all {N} nodes)\n"
                    f"tree-sparse: {density:.0%} nonzero", fontsize=11)
    ax[0].set_xlabel("node (breadth-first order)")
    ax[0].set_ylabel("node")

    im = ax[1].imshow(np.abs(Sigma), cmap="viridis", interpolation="none")
    ax[1].set_title("Covariance $\\Sigma = Q^{-1}$\n dense: every node-pair shares ancestry",
                    fontsize=11)
    ax[1].set_xlabel("node (breadth-first order)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="$|\\Sigma_{ij}|$")

    fig.tight_layout()
    out = os.path.join(HERE, "precision_vs_covariance.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}  (N={N} nodes, Q density={density:.1%})")
    return density, N


# --------------------------------------------------------------------------
# Motivation: naive tip correlation conflates ancestry with evolution.
# Even with directly observed log-expression, the Pearson correlation of the
# two genes across leaves estimates neither K's correlation nor zero reliably,
# because the tips are not independent samples. Contrasts deconfound it.
# --------------------------------------------------------------------------
def figure_confounding(n_sims=400):
    tree = Tree(balanced_newick(6, root_branch=0.5))     # 64 leaves, free root
    n = len(tree.root.get_leaves())
    leaf_names = tree.phylotree.get_leaf_names()
    C = bm_covariance(tree)
    mu = np.array([2.0, 1.6])
    sds = [0.9, 1.1]

    def sim_naive_vs_contrast(rho_true, seed0):
        naive, contrast = [], []
        for s in range(n_sims):
            rng = np.random.default_rng(seed0 + s)
            K = corr_cov(rho_true, sds)
            Z = rng.multivariate_normal(np.tile(mu, n), np.kron(C, K)).reshape(n, 2)
            naive.append(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1])
            tt = pd.DataFrame(Z, index=leaf_names, columns=["g0", "g1"])
            contrast.append(fit_bm_mv(tree, tt).correlation().iloc[0, 1])
        return np.array(naive), np.array(contrast)

    n0, c0 = sim_naive_vs_contrast(0.0, 1000)       # independent evolution
    n8, c8 = sim_naive_vs_contrast(0.8, 5000)       # correlated evolution

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for a, (naive, contrast, truth, title) in zip(
        ax,
        [(n0, c0, 0.0, "Independent evolution  ($\\rho_K = 0$)"),
         (n8, c8, 0.8, "Correlated evolution  ($\\rho_K = +0.8$)")],
    ):
        bins = np.linspace(-1, 1, 41)
        a.hist(naive, bins=bins, alpha=0.6, color="#d95f0e",
               label="naive tip correlation")
        a.hist(contrast, bins=bins, alpha=0.6, color="#2c7fb8",
               label="evolutionary (contrast) estimate")
        a.axvline(truth, color="k", ls="--", lw=1.5, label=f"true $\\rho_K = {truth:+.1f}$")
        a.set_title(title, fontsize=11)
        a.set_xlabel("correlation between the two genes")
    ax[0].set_ylabel(f"count over {n_sims} simulations")
    ax[0].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out = os.path.join(HERE, "confounding.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"[fig] wrote {out}")
    print("\n==== CONFOUNDING NUMBERS (copy into methods.md) ====")
    for tag, naive, contrast, truth in [("rho_K=0.0", n0, c0, 0.0),
                                        ("rho_K=0.8", n8, c8, 0.8)]:
        print(f"{tag}: naive  mean={naive.mean():+.3f} sd={naive.std():.3f} "
              f"range=[{naive.min():+.2f},{naive.max():+.2f}]")
        print(f"{tag}: contr. mean={contrast.mean():+.3f} sd={contrast.std():.3f} "
              f"range=[{contrast.min():+.2f},{contrast.max():+.2f}]  (truth {truth:+.1f})")


# --------------------------------------------------------------------------
# Running example: latent correlated BM seen through Poisson counts.
# --------------------------------------------------------------------------
def run_example():
    rng = np.random.default_rng(0)
    tree = Tree(balanced_newick(6, root_branch=0.5))     # 64 leaves, free root
    n = len(tree.root.get_leaves())
    leaf_names = tree.phylotree.get_leaf_names()

    # (i) Gaussian-limit identity: latent fit must match closed-form contrast MLE.
    K_g = corr_cov(0.7, [0.9, 1.2])
    mu = np.array([2.0, 1.6])
    C = bm_covariance(tree)
    Zg = rng.multivariate_normal(np.tile(mu, n), np.kron(C, K_g)).reshape(n, 2)
    import pandas as pd
    tt = pd.DataFrame(Zg, index=leaf_names, columns=["g0", "g1"])
    r_closed = fit_bm_mv(tree, tt).correlation().iloc[0, 1]
    r_latent = fit_mv_latent(tree, GaussianObservation(Zg, 1e-3), model="BM",
                             trait_names=["g0", "g1"]).correlation().iloc[0, 1]
    r_naive_g = np.corrcoef(Zg[:, 0], Zg[:, 1])[0, 1]

    # (ii) The headline run: rho_true = 0.8, recovered from Poisson counts by EM.
    K_true = corr_cov(0.8, [0.9, 1.1])
    Z = rng.multivariate_normal(np.tile(mu, n), np.kron(C, K_true)).reshape(n, 2)
    S = rng.uniform(40, 80, size=n)
    Y = rng.poisson(S[:, None] * np.exp(Z))
    obs = PoissonObservation(Y, S)

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        em = fit_mv_em(tree, obs, model="BM", trait_names=["g0", "g1"], verbose=True)
    t_em = time.perf_counter() - t0
    marg = [float(line.split("=")[-1]) for line in buf.getvalue().splitlines()
            if "marginal logL" in line]

    rho_em = em.correlation().iloc[0, 1]
    K_hat = em.covariance().values

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.4))

    ax[0, 0].scatter(Z[:, 0], Z[:, 1], s=18, alpha=0.8, color="#2c7fb8")
    ax[0, 0].set_title(f"Latent log-expression at leaves\n(true evolutionary corr = +0.80)",
                       fontsize=11)
    ax[0, 0].set_xlabel("$z_{g0}$"); ax[0, 0].set_ylabel("$z_{g1}$")

    ax[0, 1].scatter(Y[:, 0], Y[:, 1], s=18, alpha=0.8, color="#d95f0e")
    ax[0, 1].set_title("Observed Poisson counts at leaves\n(what we actually measure)",
                       fontsize=11)
    ax[0, 1].set_xlabel("$Y_{g0}$"); ax[0, 1].set_ylabel("$Y_{g1}$")

    ax[1, 0].plot(range(len(marg)), marg, "o-", color="#31a354")
    ax[1, 0].set_title("Laplace-EM marginal log-likelihood\n(monotone increase)", fontsize=11)
    ax[1, 0].set_xlabel("EM iteration"); ax[1, 0].set_ylabel("marginal $\\log p(Y\\mid\\eta)$")

    labels = ["var $g0$", "var $g1$", "cov $g0,g1$"]
    true_vals = [K_true[0, 0], K_true[1, 1], K_true[0, 1]]
    hat_vals = [K_hat[0, 0], K_hat[1, 1], K_hat[0, 1]]
    x = np.arange(3); w = 0.38
    ax[1, 1].bar(x - w / 2, true_vals, w, label="true $K$", color="#7570b3")
    ax[1, 1].bar(x + w / 2, hat_vals, w, label="EM estimate $\\hat K$", color="#e7298a")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(labels)
    ax[1, 1].set_title(f"Recovered diffusion matrix $K$\n(est. corr = {rho_em:+.2f})",
                       fontsize=11)
    ax[1, 1].legend()

    fig.tight_layout()
    out = os.path.join(HERE, "running_example.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"[fig] wrote {out}")
    print("\n==== RUNNING-EXAMPLE NUMBERS (copy into methods.md) ====")
    print(f"n_leaves           = {n}")
    print(f"Gaussian limit     : naive-tip corr = {r_naive_g:+.3f} (true 0.70), "
          f"contrast-MLE corr = {r_closed:+.3f}, "
          f"latent-fit corr = {r_latent:+.3f}, |diff| = {abs(r_closed-r_latent):.4f}")
    print(f"EM corr (true 0.80)= {rho_em:+.3f}")
    print(f"EM iters           = {em.extra['em_iters']}")
    print(f"EM runtime         = {t_em:.1f}s")
    print(f"EM marginal first/last = {marg[0]:.2f} -> {marg[-1]:.2f}")
    print(f"K_hat =\n{np.round(K_hat,3)}")
    print(f"marginal monotone increasing? {all(np.diff(marg) > -1e-6)}")


if __name__ == "__main__":
    figure_precision_vs_covariance()
    figure_confounding()
    run_example()
