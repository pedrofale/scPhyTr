"""Demonstrate / validate BM-vs-OU adaptive detection from Poisson counts.

Simulates a latent log-rate evolving on a tree (BM or OU), draws Poisson counts
through that latent rate, then runs ``detect_adaptive_counts`` (which integrates
out the latent values via a Laplace approximation) and reports recovery accuracy.
"""

import numpy as np
import pandas as pd
import ete3

import scphytr
from scphytr.tools import detect_adaptive_counts
from scphytr.utils.covariance import bm_covariance, ou_covariance


def build_tree(nwk):
    tree = scphytr.utils.Tree()
    tree.phylotree = ete3.PhyloTree(nwk, format=1)
    tree.root = tree.phylotree.get_tree_root()
    return tree


def balanced_newick(depth):
    counter = [0]

    def rec(d, is_root):
        if d == 0:
            counter[0] += 1
            return f"L{counter[0]}:1.0"
        node = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return node if is_root else node + ":1.0"

    return rec(depth, True) + ";"


def main():
    rng = np.random.default_rng(11)
    tree = build_tree(balanced_newick(5))  # 32 leaves
    names = tree.phylotree.get_leaf_names()
    n = len(names)

    n_each = 15
    bm_sigma2, ou_alpha, ou_sigma2 = 0.6, 2.0, 2.0
    base_log_rate = 3.0          # ~ exp(3) ≈ 20 expected counts at the optimum
    S = np.full(n, 40.0)         # per-species summed size factor (offset)

    C_bm = bm_covariance(tree)
    C_ou = ou_covariance(tree, ou_alpha)

    counts = {}
    truth = {}
    for k in range(n_each):
        z = rng.multivariate_normal(base_log_rate * np.ones(n), bm_sigma2 * C_bm)
        counts[f"BM_{k}"] = rng.poisson(S * np.exp(z))
        truth[f"BM_{k}"] = "BM"
    for k in range(n_each):
        z = rng.multivariate_normal(base_log_rate * np.ones(n), ou_sigma2 * C_ou)
        counts[f"OU_{k}"] = rng.poisson(S * np.exp(z))
        truth[f"OU_{k}"] = "OU"

    counts_table = pd.DataFrame(counts, index=names)
    size_factors = pd.Series(S, index=names)

    results = detect_adaptive_counts(tree, counts_table, size_factors, criterion="aicc")
    results["truth"] = [truth[g] for g in results.index]
    results["correct"] = results["selected"] == results["truth"]

    bm_acc = results.loc[results.truth == "BM", "correct"].mean()
    ou_acc = results.loc[results.truth == "OU", "correct"].mean()

    print(f"tree: n={n} leaves | Poisson counts (offset S={S[0]:.0f}, base rate ~{np.exp(base_log_rate):.0f})")
    print(f"generating: BM(sigma2={bm_sigma2}), OU(alpha={ou_alpha}, sigma2={ou_sigma2})")
    print(f"BM genes correctly called BM: {bm_acc:.0%} ({int(round(bm_acc * n_each))}/{n_each})")
    print(f"OU genes correctly called OU: {ou_acc:.0%} ({int(round(ou_acc * n_each))}/{n_each})")
    print(f"recovered OU alpha (median over OU genes): "
          f"{results.loc[results.truth == 'OU', 'alpha'].median():.2f} (true {ou_alpha})")
    print()
    cols = ["selected", "truth", "d_aicc", "alpha", "sigma2_OU", "sigma2_BM"]
    print(pd.concat([results.head(3), results.tail(3)])[cols].round(3).to_string())

    overall = (bm_acc + ou_acc) / 2
    print()
    if overall >= 0.75:
        print(f"PASS: mean recovery accuracy {overall:.0%}")
    else:
        print(f"WEAK: mean recovery accuracy {overall:.0%} (< 75%)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
