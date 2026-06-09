"""Demonstrate / validate the BM-vs-OU adaptive detection path.

Simulates traits with known generating process (BM or OU) on a tree, using the
correct dense covariances (independent of the buggy Tree.make_species_cov_matrix),
then runs ``detect_adaptive`` and reports how often each generating process is
correctly recovered by AIC/AICc model selection.
"""

import numpy as np
import pandas as pd
import ete3

import scphytr
from scphytr.tools.model_selection import detect_adaptive


def build_tree(nwk_txt):
    tree = scphytr.utils.Tree()
    tree.phylotree = ete3.PhyloTree(nwk_txt, format=1)
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


def phylo_times(tree):
    root = tree.root
    leaves = root.get_leaves()
    depth = {root: 0.0}
    for node in root.traverse("preorder"):
        for child in node.children:
            depth[child] = depth[node] + child.dist
    names = [leaf.name for leaf in leaves]
    n = len(leaves)
    T = np.array([root.dist + depth[leaf] for leaf in leaves])
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mrca = leaves[i] if i == j else tree.phylotree.get_common_ancestor(leaves[i], leaves[j])
            s[i, j] = root.dist + (depth[leaves[i]] if i == j else depth[mrca])
    return names, T, s


def bm_cov(s):
    return s.copy()  # BM covariance = shared time from the root


def ou_cov(T, s, alpha):
    return np.exp(-alpha * (T[:, None] + T[None, :] - 2.0 * s)) * (1.0 - np.exp(-2.0 * alpha * s)) / (2.0 * alpha)


def main():
    rng = np.random.default_rng(7)
    tree = build_tree(balanced_newick(6))  # 64 leaves
    names, T, s = phylo_times(tree)
    n = len(names)

    n_each = 25
    bm_sigma2, ou_alpha, ou_sigma2 = 0.5, 2.0, 2.0

    C_bm = bm_cov(s)
    C_ou = ou_cov(T, s, ou_alpha)

    columns = {}
    truth = {}
    for k in range(n_each):
        columns[f"BM_{k}"] = rng.multivariate_normal(np.zeros(n), bm_sigma2 * C_bm)
        truth[f"BM_{k}"] = "BM"
    for k in range(n_each):
        columns[f"OU_{k}"] = rng.multivariate_normal(np.zeros(n), ou_sigma2 * C_ou)
        truth[f"OU_{k}"] = "OU"

    trait_table = pd.DataFrame(columns, index=names)

    results = detect_adaptive(tree, trait_table, criterion="aicc")

    results["truth"] = [truth[t] for t in results.index]
    results["correct"] = results["selected"] == results["truth"]

    bm_acc = results.loc[results.truth == "BM", "correct"].mean()
    ou_acc = results.loc[results.truth == "OU", "correct"].mean()

    print(f"tree: n={n} leaves | generating: BM(sigma2={bm_sigma2}), "
          f"OU(alpha={ou_alpha}, sigma2={ou_sigma2})")
    print(f"BM traits correctly called BM: {bm_acc:.0%} ({int(bm_acc * n_each)}/{n_each})")
    print(f"OU traits correctly called OU: {ou_acc:.0%} ({int(ou_acc * n_each)}/{n_each})")
    print(f"recovered OU alpha (median over OU traits): "
          f"{results.loc[results.truth == 'OU', 'alpha'].median():.2f} (true {ou_alpha})")
    print()
    print("sample of results:")
    cols = ["selected", "truth", "d_aicc", "alpha", "sigma2_OU", "sigma2_BM"]
    print(pd.concat([results.head(3), results.tail(3)])[cols].round(3).to_string())

    overall = (bm_acc + ou_acc) / 2
    print()
    if overall >= 0.8:
        print(f"PASS: mean recovery accuracy {overall:.0%}")
    else:
        print(f"WEAK: mean recovery accuracy {overall:.0%} (< 80%)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
