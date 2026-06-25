"""scPhyTr vs the REAL EvoGeneX R package on the melanoma sublines (adaptive expression).

Earlier benchmarks compared scPhyTr to an EvoGeneX-*style* Gaussian-on-log surrogate. This
runs the actual EvoGeneX package (via ``evogenex_bridge.R``) and scPhyTr on the SAME tree, the
SAME two-regime painting (the paper's ``har`` chosen/background), and the SAME cells, then
compares which genes each calls adaptively shifted -- a real-data discovery comparison with no
fabricated ground truth.

  * EvoGeneX -- replicate-aware Gaussian OU on per-cell log2(1+TPM); cells are the replicates,
    a within-subline variance ``gamma`` is fit, and OU2-vs-BM is tested by AIC / LRT.
  * scPhyTr  -- per-cell Poisson/NB count model; BM vs two-regime OU on the multi-cell
    observation, selected by AIC (``analysis/melanoma/adaptive_ou2`` machinery).

Both are run in their native mode on the same underlying data, so the comparison reflects the
methods, not a re-formatting. Requires the EvoGeneX env (set ``$REVO`` to its R bin; default the
conda ``revo`` env built for this).
"""
import os
import subprocess

import numpy as np
import pandas as pd

from analysis.melanoma.load import (load_cells, load_counts, tree_leaves,
                                     load_tree, load_regimes, _REGIME_DIR)
from scphytr.inference.laplace import MultiCellPoissonObservation
from scphytr.tools.model_selection import fit_bm_counts, fit_ou_regimes_counts

OUT = os.path.dirname(__file__)
REVO = os.environ.get("REVO", os.path.expanduser("~/miniconda3/envs/revo/bin"))


def _select_genes(n_genes, regime="har", seed=0):
    """Top-``n_genes`` variable genes detected in every subline (EvoGeneX needs all leaves)."""
    Xt, genes, clone = load_cells("tpm")
    leaves = set(tree_leaves())
    keep = np.array([c in leaves for c in clone])
    Xt, clone = Xt[keep], np.asarray(clone)[keep]
    logX = np.log2(1.0 + Xt)
    sub = pd.DataFrame(logX, columns=genes); sub["__s"] = clone
    pb = sub.groupby("__s").mean()
    detected = (pb > 0).sum(0) == pb.shape[0]              # expressed in every subline
    v = logX.var(0)
    cand = [g for g in np.argsort(v)[::-1] if detected.iloc[g]][:n_genes]
    return [genes[i] for i in cand], logX, genes, clone


def run_evogenex(gene_list, logX, all_genes, clone, regime="har"):
    """Write inputs and run the EvoGeneX R bridge; return its per-gene result table."""
    work = os.path.join(OUT, "evogenex_work"); os.makedirs(work, exist_ok=True)
    # tree (floored, leaf-labelled newick) for ape/ouch
    tree = load_tree()
    nwk = os.path.join(work, "tree.nwk"); tree.phylotree.write(format=5, outfile=nwk)
    regime_csv = os.path.join(_REGIME_DIR, f"{regime}.csv")
    # long-format expression: gene, species (subline), replicate (cell), exprval
    gidx = {g: i for i, g in enumerate(all_genes)}
    rows = []
    for g in gene_list:
        col = logX[:, gidx[g]]
        rep = {}
        for cell, (s, v) in enumerate(zip(clone, col)):
            rep[s] = rep.get(s, 0) + 1
            rows.append((g, s, f"R{rep[s]}", float(v)))
    long_csv = os.path.join(work, "long.csv")
    pd.DataFrame(rows, columns=["gene", "species", "replicate", "exprval"]).to_csv(
        long_csv, index=False)
    out_csv = os.path.join(work, "evogenex_out.csv")
    rscript = os.path.join(REVO, "Rscript")
    bridge = os.path.join(OUT, "evogenex_bridge.R")
    print(f"running real EvoGeneX on {len(gene_list)} genes (regime {regime}) ...")
    subprocess.run([rscript, bridge, long_csv, nwk, regime_csv, out_csv], check=True)
    return pd.read_csv(out_csv)


def run_scphytr(gene_list, regime="har"):
    """scPhyTr adaptive (BM vs two-regime OU) on per-cell counts, AIC call per gene."""
    X, genes, clone, sf = load_counts()
    tree = load_tree()
    leaves = tree_leaves()
    regimes, n_reg = load_regimes(tree, regime)
    leaf_of = {s: i for i, s in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone])
    gpos = {g: i for i, g in enumerate(genes)}
    rows = []
    for g in gene_list:
        y = X[:, gpos[g]].astype(float)[:, None]
        obs = MultiCellPoissonObservation(y, sf, idx, len(leaves), univariate=True)
        bm = fit_bm_counts(tree, obs)
        ou2 = fit_ou_regimes_counts(tree, obs, regimes, n_reg)
        rows.append({"gene": g, "bm_aic": bm.aic(), "ou2_aic": ou2.aic(),
                     "adaptive_aic": int(ou2.aic() < bm.aic())})
    return pd.DataFrame(rows)


def main(n_genes=30, regime="har"):
    gene_list, logX, all_genes, clone = _select_genes(n_genes, regime)
    eg = run_evogenex(gene_list, logX, all_genes, clone, regime).set_index("gene")
    sp = run_scphytr(gene_list, regime).set_index("gene")
    df = sp.join(eg, lsuffix="_sp", rsuffix="_eg")
    out_csv = os.path.join(OUT, "evogenex_compare.csv")
    df.to_csv(out_csv)
    print(f"\nwrote {out_csv}")

    a_sp = df["adaptive_aic_sp"] == 1
    a_eg = df["adaptive_aic_eg"] == 1
    print("\n===== scPhyTr vs REAL EvoGeneX: adaptive discoveries (melanoma, "
          f"regime {regime}, {len(df)} genes) =====")
    print(f"  adaptive calls: scPhyTr {a_sp.sum()}, EvoGeneX {a_eg.sum()}")
    print(f"  both adaptive: {(a_sp & a_eg).sum()};  neither: {(~a_sp & ~a_eg).sum()};  "
          f"scPhyTr-only: {(a_sp & ~a_eg).sum()};  EvoGeneX-only: {(~a_sp & a_eg).sum()}")
    agree = (a_sp == a_eg).mean()
    print(f"  binary-call agreement: {agree:.0%}")
    return df


if __name__ == "__main__":
    main()
