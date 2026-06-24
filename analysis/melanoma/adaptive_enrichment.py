"""Compare scPhyTr's adaptive (OU-2) genes to the paper's Wnt-signaling claim.

The paper found that EvoGeneX adaptive genes enrich for KEGG Wnt signaling
(canonical = proliferation/sensitive, non-canonical = invasion/resistant). This
tests the same on scPhyTr's OU-2 calls: a Fisher exact test of Wnt-pathway
membership among adaptive vs non-adaptive genes in a scan, plus the list of
adaptive Wnt genes. Two modes:

  python -m analysis.melanoma.adaptive_enrichment <scan.csv>   # test an existing scan
  python -m analysis.melanoma.adaptive_enrichment --targeted [regime]
        # scan the data's Wnt genes + a size-matched random control, then test
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

_HERE = os.path.dirname(os.path.abspath(__file__))
WNT = os.path.join(_HERE, "..", "..", "data", "external", "kegg_wnt_mmu04310.txt")


def wnt_genes():
    with open(WNT) as f:
        return {ln.strip() for ln in f if ln.strip()}


def enrich(df, label=""):
    """Fisher exact: adaptive (OU-2) x Wnt-pathway membership over scanned genes."""
    wnt = wnt_genes()
    is_wnt = df.index.to_series().isin(wnt)
    adaptive = df["adaptive"].astype(bool)
    a = int((adaptive & is_wnt).sum())          # adaptive & Wnt
    b = int((adaptive & ~is_wnt).sum())          # adaptive & not Wnt
    c = int((~adaptive & is_wnt).sum())          # not adaptive & Wnt
    d = int((~adaptive & ~is_wnt).sum())
    table = [[a, b], [c, d]]
    odds, p = fisher_exact(table, alternative="greater")
    print(f"\n=== {label} ===")
    print(f"scanned {len(df)} genes; {is_wnt.sum()} Wnt, {adaptive.sum()} adaptive (OU-2)")
    print(f"adaptive Wnt genes: {a}  |  contingency [[A&W,A&~W],[~A&W,~A&~W]] = {table}")
    print(f"Fisher exact (adaptive enriched for Wnt): odds={odds:.2f}, p={p:.3f}")
    aw = sorted(df.index[(adaptive & is_wnt)])
    print(f"adaptive Wnt genes: {aw}")
    print(f"all adaptive genes: {sorted(df.index[adaptive])}")
    return p


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--targeted":
        from analysis.melanoma.adaptive_ou2 import scan
        regime = sys.argv[2] if len(sys.argv) > 2 else "har"
        rng = np.random.default_rng(0)
        wnt = wnt_genes()
        # all data gene symbols
        from analysis.melanoma.load import load_counts
        _, genes, _, _ = load_counts()
        sym = [g.split("_")[-1] for g in genes]
        wnt_in = [s for s in sym if s in wnt]
        control = list(rng.choice([s for s in sym if s not in wnt],
                                  size=min(120, len(wnt_in) * 3), replace=False))
        targets = sorted(set(wnt_in) | set(control))
        print(f"targeted scan: {len(wnt_in)} Wnt + {len(control)} control genes, regime '{regime}'")
        df = scan(regime, dispersion="nb", verbose=False, gene_symbols=targets)
        out = os.path.join(_HERE, "figures", f"adaptive_targeted_{regime}.csv")
        df.to_csv(out)
        enrich(df, f"targeted Wnt-vs-control, regime '{regime}'")
    else:
        path = sys.argv[1]
        df = pd.read_csv(path, index_col=0)
        enrich(df, os.path.basename(path))


if __name__ == "__main__":
    main()
