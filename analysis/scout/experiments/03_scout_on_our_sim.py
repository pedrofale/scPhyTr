"""Run the ORIGINAL SCOUT on OUR simulated data, and score it against our own baseline.

Pipeline: simulate from the sparse-OU model -> export to SCOUT's input format -> invoke the real R
package (`scripts/scout`) -> score both SCOUT and our idealised per-gene baseline against truth.

This is the harness the hierarchical model will eventually be measured in. Note that SCOUT is given
the TRUE regime labels (the adaptive landscape it requires), which is the fair setting for a
gene-level classification comparison.

    python -m analysis.scout.experiments.03_scout_on_our_sim
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from scphytr.modes._tree import Tree
from scphytr.modes.simulate import simulate_tips, leaf_regimes
from analysis.scout.scout_io import write_scout_dataset
from scphytr.modes.baseline import classify_genes
from analysis.scout.scout_preprocess import scout_preprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, SIGMA, N_PER_CLASS = 0.75, 0.75, 30


def two_clades(tree, lo=0.15, hi=0.40):
    sets = tree.leaf_sets(); n = tree.n_leaves
    cand = [v for v in range(tree.n_nodes)
            if tree.parent[v] >= 0 and lo * n <= len(sets[v]) <= hi * n]
    a = cand[0]; la = set(sets[a].tolist())
    for b in cand[1:]:
        if not (la & set(sets[b].tolist())):
            return a, b
    raise ValueError("no two disjoint clades")


def simulate_panel(tree, seed=0, counts=True, depth=500.0):
    """BM1 / OU1 / OUX genes; optionally pushed through a Poisson count layer like SCOUT's sims."""
    rng = np.random.default_rng(seed)
    G = N_PER_CLASS
    zero = np.zeros((tree.n_nodes, G))
    Y_bm = simulate_tips(tree, 1e-12, SIGMA, rng.normal(size=G), zero, rng=rng, root="fixed")
    Y_ou1 = simulate_tips(tree, ALPHA, SIGMA, rng.normal(size=G), zero, rng=rng, root="stationary")
    b1, b2 = two_clades(tree)
    delta = np.zeros((tree.n_nodes, G))
    delta[b1] = rng.normal(0, 2.0, size=G); delta[b2] = rng.normal(0, 2.0, size=G)
    Y_ou3 = simulate_tips(tree, ALPHA, SIGMA, rng.normal(size=G), delta, rng=rng, root="stationary")
    z = np.zeros(tree.n_nodes, bool); z[b1] = z[b2] = True
    X = np.concatenate([Y_bm, Y_ou1, Y_ou3], axis=1)
    truth = np.array(["BM1"] * G + ["OU1"] * G + ["OUX"] * G, dtype=object)
    names = ([f"BM1_{i}" for i in range(G)] + [f"OU1_{i}" for i in range(G)]
             + [f"OUM_{i}" for i in range(G)])
    if counts:                       # latent -> rate -> Poisson counts (a SymSim-like decoder)
        lam = depth * np.exp(X - X.mean(0)) / np.exp(X - X.mean(0)).mean(0)
        Y = rng.poisson(lam).astype(float)
    else:
        Y = X
    return Y, truth, names, leaf_regimes(tree, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--method", default="SM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cores", type=int, default=4)
    args = ap.parse_args()

    tree = Tree.balanced(args.n, branch_length=1.0).scale_height(1.0)
    Y, truth, names, lr = simulate_panel(tree, seed=args.seed)
    work = os.path.join(HERE, "results", f"oursim_n{args.n}_s{args.seed}")
    os.makedirs(work, exist_ok=True)
    counts_csv, tree_nwk = write_scout_dataset(os.path.join(work, "sim"), Y, tree,
                                               leaf_regime=lr, gene_names=names)
    print(f"simulated {tree.n_leaves} tips x {len(names)} genes -> {counts_csv}")

    print(f"running ORIGINAL SCOUT (method={args.method}) ...", flush=True)
    r = subprocess.run([os.path.join(HERE, "scripts", "scout"), "--counts", counts_csv,
                        "--tree", tree_nwk, "--out", work, "--regimes", "BM1,OU1,OUM",
                        "--method", args.method, "--cores", str(args.cores),
                        "--testid", "oursim"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit("SCOUT failed")

    hits = glob.glob(os.path.join(work, "*annotated_best_fit.csv"))
    d = pd.read_csv(hits[0])
    d["truth"] = d.gene_name.str.rsplit("_", n=1).str[0].replace({"OUM": "OUX"})
    d["called"] = d.model.replace({"OUM": "OUX"})
    acc_scout = float((d.truth == d.called).mean())

    X, t = scout_preprocess(Y, tree, normalize=True, smoothing_k=8)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    call, _, _ = classify_genes(Z, t, leaf_regime=lr)
    acc_ours = float(np.mean(call == truth))

    print(f"\n== our simulation, n={tree.n_leaves} tips, {len(names)} genes ==")
    print(f"  ORIGINAL SCOUT ({args.method}) : accuracy {acc_scout:.3f}")
    print(f"  our idealised baseline (k=8)  : accuracy {acc_ours:.3f}")
    print("\nSCOUT confusion:"); print(pd.crosstab(d.truth, d.called).to_string())
    pd.DataFrame(dict(gene=names, truth=truth, ours=call)).to_csv(
        os.path.join(work, "our_baseline_calls.csv"), index=False)
    print(f"\n[outputs in {work}]")


if __name__ == "__main__":
    main()
