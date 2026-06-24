"""Transcriptome-wide OU-2 adaptive scan (per-cell NB) + fair Wnt enrichment.

Scans every gene detected in all sublines (the detection-matched universe, so the
Wnt-vs-rest test is fair) with the batched NB engine, AIC-selects BM/OU1/OU2, and
Fisher-tests whether OU-2 (adaptive) genes are enriched for KEGG Wnt signaling --
the paper's headline claim, now at transcriptome scale.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from analysis.melanoma.load import load_counts, tree_leaves, load_tree, load_regimes
from analysis.melanoma.batched_scan import fit_adaptive
from analysis.melanoma.adaptive_enrichment import wnt_genes

_FIG = os.path.join(os.path.dirname(__file__), "figures")


def main(regime="har"):
    X, genes, clone, sf = load_counts()
    leaves = tree_leaves(); leaf_of = {n: k for k, n in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone]); nL = len(leaves)
    tree = load_tree(); regimes, n_reg = load_regimes(tree, regime)

    # universe: genes detected (>0) in every subline
    det = pd.DataFrame((X > 0).astype(float)); det["__c"] = idx
    detmin = det.groupby("__c").mean().min(0).values
    cols = list(np.where(detmin > 0)[0])
    sym = np.array([genes[g].split("_")[-1] for g in cols])
    print(f"universe: {len(cols)} genes detected in all {nL} sublines")

    t = time.time()
    df = fit_adaptive(tree, X, idx, nL, sf, cols, regimes, n_reg)
    df.index = sym
    print(f"scan: {time.time()-t:.0f}s  adaptive(OU2)={int(df['adaptive'].sum())} "
          f"({df['adaptive'].mean()*100:.0f}%)")
    os.makedirs(_FIG, exist_ok=True)
    df.to_csv(os.path.join(_FIG, f"transcriptome_{regime}.csv"))

    # fair Wnt enrichment within the universe
    wnt = wnt_genes()
    isw = df.index.to_series().isin(wnt); ad = df["adaptive"].astype(bool)
    a = int((ad & isw).sum()); b = int((ad & ~isw).sum())
    c = int((~ad & isw).sum()); d = int((~ad & ~isw).sum())
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    print(f"\nWnt enrichment among adaptive (detection-matched universe):")
    print(f"  Wnt in universe: {int(isw.sum())}; adaptive Wnt: {a}")
    print(f"  adaptive rate: Wnt {a/max(isw.sum(),1)*100:.0f}% vs rest "
          f"{b/max((~isw).sum(),1)*100:.0f}%")
    print(f"  Fisher (adaptive enriched for Wnt): odds={odds:.2f}, p={p:.4f}")
    print(f"  adaptive Wnt genes: {sorted(df.index[ad & isw])[:25]}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "har")
