"""Figures for phylogenetic factor analysis (PFA): docs/03_factor_analysis.md.

Demonstrates the central claims:
  (1) Plain factor analysis identifies factors only up to a rotation; with a
      phylogeny *and heterogeneous evolutionary dynamics* the factors become
      individually identifiable.
  (2) The dynamics of each factor (BM vs OU, and the BM rate) are recovered by
      model selection over the per-factor configuration.
  (3) The phylogeny-aware factor scores are accurate trajectories on the tree.

Outputs docs/figures/factor_analysis.png and prints the numbers used in the
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
    simulate_pfa, fit_factor_analysis, detect_factor_dynamics, subspace_error)

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


def match_columns(W_est, W_true):
    """Greedily match estimated to true columns by |cosine|; return (perm, cos)."""
    a = W_est / np.linalg.norm(W_est, axis=0, keepdims=True)
    b = W_true / np.linalg.norm(W_true, axis=0, keepdims=True)
    M = np.abs(a.T @ b)
    used, perm, cos = set(), [], []
    for j in range(b.shape[1]):
        order = [(M[i, j] if i not in used else -1.0) for i in range(a.shape[1])]
        i = int(np.argmax(order))
        used.add(i)
        perm.append(i)
        cos.append(M[i, j])
    return perm, np.array(cos)


def main():
    rng = np.random.default_rng(0)
    tree = Tree(balanced_newick(6, root_branch=0.5))      # 64 leaves
    leaf_names = [l.name for l in tree.root.get_leaves()]
    n = len(leaf_names)
    p, k = 10, 2

    # Two real gene programs with DIFFERENT evolutionary dynamics:
    # factor 0 drifts (BM); factor 1 is under stabilizing selection (OU).
    W_true = rng.standard_normal((p, k))
    W_true[:, 0] *= 1.3
    dynamics = [("BM", 1.0), ("OU", 4.0, 1.0)]
    true_labels = ["BM", "OU"]
    Y, X_true, _ = simulate_pfa(tree, W_true, dynamics, noise_sd=0.3, mu=1.0, seed=3)

    # Phylogenetic FA with per-factor dynamics selection (identifies factors).
    best, fits = detect_factor_dynamics(tree, Y, k=k, criterion="aic", restarts=2, seed=0)
    # Naive (phylogeny-ignorant) factor analysis.
    naive = fit_factor_analysis(Y, row_cov=None, k=k, restarts=2, seed=0)

    permP, cosP = match_columns(best.W, W_true)
    permN, cosN = match_columns(naive.W, W_true)
    sub_err = subspace_error(best.W, W_true)

    # Phylogeny-aware factor scores (BM-equivalent fit gives the tree-smoothed
    # posterior trajectories); used for panel (D).
    fa_phylo = fit_factor_analysis(Y, row_cov=bm_covariance(tree), k=k, restarts=2, seed=0)
    phylo_scores = fa_phylo.factor_scores()

    print("\n=== Phylogenetic factor analysis demonstration ===")
    print(f"tree: {n} leaves, p={p} genes, k={k} factors")
    print(f"true dynamics: {true_labels}; selected config: {list(best.config)}")
    print(best.summary())
    print(f"per-factor |cosine| to truth -- PFA(dyn): {np.round(cosP,3)} | naive: {np.round(cosN,3)}")
    print(f"loading subspace error (both): {sub_err:.3f}")
    aic_by_cfg = {f.config: f.aic() for f in fits}
    for cfg, a in sorted(aic_by_cfg.items(), key=lambda kv: kv[1]):
        print(f"   AIC {cfg}: {a:.1f}")

    # ---- Figure ---------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6))

    # (A) per-factor recovery: identifiability
    x = np.arange(k)
    ax[0, 0].bar(x - 0.18, cosP, width=0.36, label="phylo + dynamics", color="#2c7fb8")
    ax[0, 0].bar(x + 0.18, cosN, width=0.36, label="naive FA", color="#d95f0e")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([f"factor {j}\n(true {true_labels[j]})" for j in range(k)])
    ax[0, 0].set_ylabel("|cosine| of recovered vs true loading")
    ax[0, 0].set_ylim(0, 1.05)
    ax[0, 0].axhline(1.0, ls=":", c="grey", lw=1)
    ax[0, 0].set_title("(A) Individual factor identifiability\n(subspace identical: "
                       f"err={sub_err:.2f})", fontsize=11)
    ax[0, 0].legend(fontsize=9)

    # (B) AIC over the 2^k dynamics configurations
    cfgs = sorted(aic_by_cfg, key=lambda c: aic_by_cfg[c])
    vals = [aic_by_cfg[c] - min(aic_by_cfg.values()) for c in cfgs]
    labels = ["/".join(c) for c in cfgs]
    colors = ["#31a354" if c == best.config else "#bdbdbd" for c in cfgs]
    ax[0, 1].bar(range(len(cfgs)), vals, color=colors)
    ax[0, 1].set_xticks(range(len(cfgs)))
    ax[0, 1].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax[0, 1].set_ylabel(r"$\Delta$AIC vs best")
    ax[0, 1].set_title("(B) Per-factor dynamics selection\n(true = BM/OU)", fontsize=11)

    # (C) recovered rate / alpha per factor
    rates = best.rates()
    ax[1, 0].bar(x - 0.18, rates / rates.max(), width=0.36, color="#756bb1",
                 label="rate (norm.)")
    alph = np.nan_to_num(best.alphas, nan=0.0)
    ax[1, 0].bar(x + 0.18, alph / (alph.max() if alph.max() > 0 else 1), width=0.36,
                 color="#dd1c77", label=r"$\alpha$ (norm.)")
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels([f"factor {j}\n({best.config[j]})" for j in range(k)])
    ax[1, 0].set_title("(C) Recovered evolutionary parameters", fontsize=11)
    ax[1, 0].legend(fontsize=9)

    # (D) phylo factor scores vs truth (aligned), per factor
    permp, _ = match_columns(fa_phylo.W, W_true)
    for j in range(k):
        col = fa_phylo.W[:, permp[j]]
        sign = np.sign(np.dot(col, W_true[:, j]) + 1e-12)
        sc = sign * phylo_scores[:, permp[j]]
        ax[1, 1].scatter(X_true[:, j], sc, s=14, alpha=0.7,
                         label=f"factor {j} ({true_labels[j]})")
    lims = [X_true.min() - 0.5, X_true.max() + 0.5]
    ax[1, 1].plot(lims, lims, ls=":", c="grey", lw=1)
    ax[1, 1].set_xlabel("true factor value at leaf")
    ax[1, 1].set_ylabel("inferred (phylo) factor score")
    ax[1, 1].set_title("(D) Phylogenetic factor scores recover trajectories", fontsize=11)
    ax[1, 1].legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(HERE, "factor_analysis.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[fig] wrote {out}")


if __name__ == "__main__":
    main()
