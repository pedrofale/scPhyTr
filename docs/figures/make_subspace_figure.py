"""Figure for docs/03_factor_analysis.md S3.1: the loading *subspace* is
phylogeny-invariant (in expectation), but individual factors are not identified
under shared dynamics.

Simulates PFA data under a single shared Brownian process (so the only thing
that differs between the two fits is the row covariance), fits naive and
phylogenetic factor analysis, and over many seeds reports:

  (A) the loading-*subspace* error (Grassmann distance) to the truth -- naive and
      phylo coincide: the tree does not change which subspace the factors span;
  (B) the best per-factor |cosine| to the true loadings -- both are limited by
      the rotational ambiguity of factor analysis under shared dynamics, which
      motivates the heterogeneous-dynamics identifiability of S3.3.

Outputs docs/figures/subspace_invariance.png and prints the numbers used in the
write-up. Run inside the `scphytr` conda env. Deterministic.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance
from scphytr.tools.factor_analysis import (
    simulate_pfa, fit_factor_analysis, subspace_error)

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


def best_factor_cosines(W_est, W_true):
    """Greedy best |cosine| match of estimated to true loading columns."""
    a = W_est / np.linalg.norm(W_est, axis=0, keepdims=True)
    b = W_true / np.linalg.norm(W_true, axis=0, keepdims=True)
    M = np.abs(a.T @ b)
    used, cos = set(), []
    for j in range(b.shape[1]):
        order = [(M[i, j] if i not in used else -1.0) for i in range(a.shape[1])]
        i = int(np.argmax(order))
        used.add(i)
        cos.append(M[i, j])
    return np.array(cos)


def main():
    tree = Tree(balanced_newick(6, root_branch=0.5))      # 64 leaves
    C = bm_covariance(tree)
    p, k = 10, 2
    R = 25

    sub_naive, sub_phylo = [], []
    cos_naive, cos_phylo = [], []
    for seed in range(R):
        rng = np.random.default_rng(seed)
        W_true = rng.standard_normal((p, k))
        # Single shared Brownian process for both factors: the ONLY difference
        # between the two fits is the row covariance (tree C vs identity).
        dynamics = [("BM", 1.0), ("BM", 1.0)]
        Y, X_true, _ = simulate_pfa(tree, W_true, dynamics, noise_sd=0.3, mu=1.0, seed=seed)

        naive = fit_factor_analysis(Y, row_cov=None, k=k, restarts=2, seed=0)
        phylo = fit_factor_analysis(Y, row_cov=C, k=k, restarts=2, seed=0)

        sub_naive.append(subspace_error(naive.W, W_true))
        sub_phylo.append(subspace_error(phylo.W, W_true))
        cos_naive.append(best_factor_cosines(naive.W, W_true))
        cos_phylo.append(best_factor_cosines(phylo.W, W_true))

    sub_naive = np.array(sub_naive)
    sub_phylo = np.array(sub_phylo)
    cos_naive = np.concatenate(cos_naive)
    cos_phylo = np.concatenate(cos_phylo)

    print("\n=== S3.1 loading-subspace invariance (shared BM dynamics) ===")
    print(f"tree: 64 leaves, p={p}, k={k}, {R} seeds")
    print(f"subspace error to truth   naive: {sub_naive.mean():.3f} +/- {sub_naive.std():.3f}"
          f"   phylo: {sub_phylo.mean():.3f} +/- {sub_phylo.std():.3f}")
    print(f"  |difference| per seed: mean {np.abs(sub_naive - sub_phylo).mean():.3f} "
          f"(the two coincide)")
    print(f"best per-factor |cosine|  naive: {cos_naive.mean():.3f}"
          f"   phylo: {cos_phylo.mean():.3f}   (both rotation-limited under shared dynamics)")

    # ----- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # (A) subspace error: naive vs phylo coincide
    ax[0].scatter(sub_naive, sub_phylo, s=34, color="#2c7fb8", alpha=0.8, zorder=3)
    lim = [0, max(sub_naive.max(), sub_phylo.max()) * 1.15 + 1e-9]
    ax[0].plot(lim, lim, ls=":", c="grey", lw=1.2, label="y = x")
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel("subspace error to truth -- naive FA")
    ax[0].set_ylabel("subspace error to truth -- phylo FA")
    ax[0].legend(fontsize=9)
    ax[0].set_title("(A) The loading subspace is phylogeny-invariant\n"
                    f"naive {sub_naive.mean():.2f} vs phylo {sub_phylo.mean():.2f} "
                    "(coincide over 25 seeds)", fontsize=11)

    # (B) per-factor identifiability: both limited under shared dynamics
    ax[1].boxplot([cos_naive, cos_phylo], tick_labels=["naive FA", "phylo FA"],
                  widths=0.5, patch_artist=True,
                  boxprops=dict(facecolor="#deebf7"),
                  medianprops=dict(color="#08519c"))
    ax[1].axhline(1.0, ls=":", c="grey", lw=1)
    ax[1].set_ylabel("best per-factor |cosine| to true loading")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("(B) Individual factors are NOT identified\n"
                    "under shared dynamics (rotation ambiguity -> S3.3)", fontsize=11)

    fig.tight_layout()
    out = os.path.join(HERE, "subspace_invariance.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[fig] wrote {out}")


if __name__ == "__main__":
    main()
