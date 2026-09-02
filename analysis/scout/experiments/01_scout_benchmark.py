"""Reproduce SCOUT's headline benchmark (Fig 2B) in our own code -- the per-gene baseline.

SCOUT's setting: 50 genes for each of BM1 / OU1 / OU3, trees from 32 to 4096 leaves,
alpha = sigma = 0.75, 3-way classification accuracy under min-AICc model selection, with the regime
painting SUPPLIED. Their "ground truth" arm uses the continuous OU traits before the count
transform; that is the arm we match here (no count layer yet).

This gives us a like-for-like per-gene baseline implemented in our own likelihood, with no R and no
TedSim -- the reference the hierarchical sparse model must beat.

    python -m analysis.scout.experiments.01_scout_benchmark
"""
import argparse
import time

import numpy as np
import pandas as pd

from scphytr.modes._tree import Tree
from scphytr.modes.simulate import simulate_tips, leaf_regimes
from scphytr.modes.baseline import classify_genes

ALPHA, SIGMA = 0.75, 0.75
N_PER_CLASS = 50


def two_event_branches(tree, min_frac=0.15, max_frac=0.40):
    """Two disjoint (non-nested) clades of moderate size -> a 3-regime landscape."""
    sets = tree.leaf_sets()
    n = tree.n_leaves
    cand = [v for v in range(tree.n_nodes)
            if tree.parent[v] >= 0 and min_frac * n <= len(sets[v]) <= max_frac * n]
    if len(cand) < 2:
        raise ValueError("no candidate clades")
    a = cand[0]
    la = set(sets[a].tolist())
    for b in cand[1:]:
        if not (la & set(sets[b].tolist())):
            return a, b
    raise ValueError("could not find two disjoint clades")


def simulate_panel(tree, seed=0):
    """50 BM1 + 50 OU1 + 50 OU3 genes; returns (Y, truth labels, leaf regimes for the OU3 painting)."""
    rng = np.random.default_rng(seed)
    G = N_PER_CLASS
    zero = np.zeros((tree.n_nodes, G))

    # BM1: neutral drift, root state free
    Y_bm = simulate_tips(tree, 1e-12, SIGMA, rng.normal(size=G), zero, rng=rng, root="fixed")
    # OU1: single global optimum
    Y_ou1 = simulate_tips(tree, ALPHA, SIGMA, rng.normal(size=G), zero, rng=rng, root="stationary")
    # OU3: two clades shift to their own optima (all genes respond, as in SCOUT's OUx)
    b1, b2 = two_event_branches(tree)
    delta = np.zeros((tree.n_nodes, G))
    delta[b1] = rng.normal(0.0, 2.0, size=G)
    delta[b2] = rng.normal(0.0, 2.0, size=G)
    Y_ou3 = simulate_tips(tree, ALPHA, SIGMA, rng.normal(size=G), delta, rng=rng, root="stationary")

    z = np.zeros(tree.n_nodes, dtype=bool); z[b1] = z[b2] = True
    Y = np.concatenate([Y_bm, Y_ou1, Y_ou3], axis=1)
    truth = np.array(["BM1"] * G + ["OU1"] * G + ["OUX"] * G, dtype=object)
    return Y, truth, leaf_regimes(tree, z)


def run(sizes, seeds=(0, 1, 2)):
    rows = []
    for n in sizes:
        for seed in seeds:
            tree = Tree.balanced(n, branch_length=1.0).scale_height(1.0)
            Y, truth, lr = simulate_panel(tree, seed=seed)
            t0 = time.time()
            call, fits, gap = classify_genes(Y, tree, leaf_regime=lr)
            acc = float(np.mean(call == truth))
            per = {c: float(np.mean(call[truth == c] == c)) for c in ("BM1", "OU1", "OUX")}
            rows.append(dict(n_leaves=n, seed=seed, accuracy=acc, secs=round(time.time() - t0, 1),
                             **{f"recall_{k}": v for k, v in per.items()}))
            print(f"  n={n:5d} seed={seed}  acc={acc:.3f}  "
                  f"(BM1 {per['BM1']:.2f} / OU1 {per['OU1']:.2f} / OUX {per['OUX']:.2f})  "
                  f"[{rows[-1]['secs']}s]", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="32,64,128,256,512")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/scout_baseline.csv")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"SCOUT-style per-gene baseline (alpha={ALPHA}, sigma={SIGMA}, "
          f"{N_PER_CLASS} genes/class, regimes SUPPLIED)")
    df = run(sizes, seeds)
    df.to_csv(args.out, index=False)
    print("\n== mean accuracy by tree size ==")
    print(df.groupby("n_leaves")["accuracy"].agg(["mean", "std"]).round(3).to_string())
    print(f"\n[wrote {args.out}]")
    print("SCOUT reports 0.55 [0.47,0.62] at 32 leaves (log-norm arm) and near-ground-truth by ~128.")


if __name__ == "__main__":
    main()
