"""Validate the heritable-vs-niche variance decomposition (tl.decompose_variance).

Using scPhyTr's native spatial simulator (``ph.simulate_spatial_panel``: BM coordinates + additive
phylogenetic-BM and spatial-field expression, NB counts) with a *known* per-gene split, we check:

 1. CORRECTNESS  -- on Gaussian observations the joint Laplace decomposition matches an independent
    closed-form REML reference (validates the engine).
 2. RECOVERY     -- the recovered ``frac_heritable`` lands on the diagonal vs the planted truth
    across heritable / niche / mixed genes.
 3. BASELINE     -- a tree-only BM rate (``tl.estimate_rate``) misattributes spatial (niche) genes
    as fast-evolving; the decomposition does not.
 4. IDENTIFIABILITY -- accuracy vs clonal intermixing: the phylo/niche split is well recovered when
    cells intermix in space and degrades gracefully toward pure-lineage coordinates.
"""
import os
import numpy as np
import pandas as pd

import scphytr as ph
from scphytr.utils.tree import Tree
from scphytr.observation_models import SubclonalObservation
from scphytr.inference.spatial_decomposition import (decompose, reml_gaussian_reference,
                                                     GaussianLeafObservation)
from ete3 import Tree as ETree

OUT = os.path.dirname(__file__)


def _tree(n=60, seed=0):
    et = ETree(); et.populate(n, random_branches=True)
    for nd in et.traverse():
        if not nd.is_root():
            nd.dist = round(max(float(nd.dist), 0.1), 3)
    for i, l in enumerate(et.get_leaves()):
        l.name = f"c{i}"
    tw = Tree(); tw.phylotree = et; tw.root = et.get_tree_root()
    return tw


def _panel(n_each=8):
    """Per-gene planted (v_phylo, v_space): heritable / niche / mixed blocks."""
    v_ph = np.r_[np.full(n_each, 2.0), np.full(n_each, 0.1), np.full(n_each, 1.0)]
    v_sp = np.r_[np.full(n_each, 0.1), np.full(n_each, 2.0), np.full(n_each, 1.0)]
    grp = ["heritable"] * n_each + ["niche"] * n_each + ["mixed"] * n_each
    names = ([f"her{i}" for i in range(n_each)] + [f"nic{i}" for i in range(n_each)]
             + [f"mix{i}" for i in range(n_each)])
    return v_ph, v_sp, grp, names


def correctness(n=50, seed=0):
    """Gaussian decompose vs independent REML reference (scale-based frac_heritable)."""
    tree = _tree(n, seed)
    v_ph, v_sp, grp, names = _panel(n_each=3)
    A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0, dispersion=[30.] * len(names),
                                  n_cells=1, mean_size=500, gene_names=names, seed=seed, intermixing=0.6)
    ph.pp.setup_anndata(A, tree); ph.pp.spatial_neighbors(A, n_neighbors=8)
    Qs = A.uns["spatial_graph"]["precision"]; leaves = tree.phylotree.get_leaf_names()
    Z = A.uns["true_latent"]
    print("== (1) correctness: Gaussian decompose vs REML reference ==")
    rows = []
    for g, nm in enumerate(names):
        y = Z[:, g] + np.random.default_rng(g).normal(0, 0.25, len(leaves))
        d = decompose(tree, GaussianLeafObservation(y - y.mean(), noise=0.0625), Qs, include_residual=False)
        r = reml_gaussian_reference(tree, y, Qs)
        rows.append((nm, d.frac_heritable_scale, r["frac_heritable"]))
        print(f"  {nm:5s}: decompose {d.frac_heritable_scale:.2f}  REML {r['frac_heritable']:.2f}")
    err = np.mean([abs(a - b) for _, a, b in rows])
    print(f"  mean |decompose - REML| frac = {err:.3f}  (engine matches the closed-form reference)")
    return err


def recovery(n=60, n_each=8, intermixing=0.6, dispersion=30.0, seed=0):
    """Count recovery + tree-only baseline across heritable/niche/mixed genes."""
    tree = _tree(n, seed)
    v_ph, v_sp, grp, names = _panel(n_each)
    A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0,
                                  dispersion=[dispersion] * len(names), n_cells=1, mean_size=500,
                                  gene_names=names, seed=seed, intermixing=intermixing)
    ph.pp.setup_anndata(A, tree)
    ph.tl.decompose_variance(A, dispersion=dispersion)        # heritable/niche split
    ph.tl.estimate_rate(A, dispersion=dispersion)             # tree-only baseline (BM rate)
    A.var["group"] = grp
    A.var["true_frac"] = v_ph / (v_ph + v_sp)
    df = A.var[["group", "true_frac", "frac_heritable", "v_phylo", "v_space", "rate"]].copy()
    df.to_csv(os.path.join(OUT, "spatial_decomposition.csv"))

    print("\n== (2) recovery: frac_heritable vs planted ==")
    for grp_name in ["heritable", "niche", "mixed"]:
        d = df[df.group == grp_name]
        print(f"  {grp_name:9s}: recovered frac {d.frac_heritable.mean():.2f} ± {d.frac_heritable.std():.2f}"
              f"  (true {d.true_frac.iloc[0]:.2f})")
    mae = (df.frac_heritable - df.true_frac).abs().mean()
    print(f"  mean |recovered - true| frac = {mae:.3f}")

    print("\n== (3) baseline: a tree-only BM rate misattributes niche genes ==")
    her_rate = df[df.group == "heritable"]["rate"].mean()
    nic_rate = df[df.group == "niche"]["rate"].mean()
    print(f"  tree-only BM rate: heritable {her_rate:.2f} vs niche {nic_rate:.2f}  "
          f"-> niche looks {'FASTER' if nic_rate > her_rate else 'slower'} (confounded); "
          f"decompose frac heritable {df[df.group=='heritable'].frac_heritable.mean():.2f} "
          f"vs niche {df[df.group=='niche'].frac_heritable.mean():.2f} (correct)")
    return df


def identifiability(n=60, n_each=6, levels=(0.0, 0.2, 0.4, 0.7, 1.0), dispersion=30.0, seed=0):
    """Recovery error of frac_heritable vs clonal intermixing."""
    v_ph, v_sp, grp, names = _panel(n_each)
    true_frac = v_ph / (v_ph + v_sp)
    rows = []
    print("\n== (4) identifiability: recovery vs intermixing ==")
    for rho in levels:
        tree = _tree(n, seed)
        A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0,
                                      dispersion=[dispersion] * len(names), n_cells=1, mean_size=500,
                                      gene_names=names, seed=seed, intermixing=rho)
        ph.pp.setup_anndata(A, tree)
        ph.tl.decompose_variance(A, dispersion=dispersion)
        mae = float(np.mean(np.abs(A.var["frac_heritable"].values - true_frac)))
        rows.append({"intermixing": rho, "frac_mae": mae})
        print(f"  intermixing {rho:.1f}: mean |recovered - true| frac = {mae:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "spatial_decomposition_identifiability.csv"), index=False)
    return df


def figure(rec_df=None, idf_df=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rec_df is None:
        rec_df = pd.read_csv(os.path.join(OUT, "spatial_decomposition.csv"), index_col=0)
    if idf_df is None:
        idf_df = pd.read_csv(os.path.join(OUT, "spatial_decomposition_identifiability.csv"))
    col = {"heritable": "#2c7fb8", "niche": "#e45756", "mixed": "#54a24b"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (A) recovered vs true frac_heritable
    for grp in col:
        d = rec_df[rec_df.group == grp]
        ax[0].scatter(d.true_frac, d.frac_heritable, color=col[grp], s=40, label=grp, edgecolor="k", lw=0.3)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("true frac heritable"); ax[0].set_ylabel("recovered frac heritable")
    ax[0].set_title("(A) Recovery of the heritable/niche split"); ax[0].legend(fontsize=8)
    # (B) baseline confound: tree-only rate vs decomposition
    x = np.arange(3); w = 0.36; groups = ["heritable", "niche", "mixed"]
    rate = [rec_df[rec_df.group == g]["rate"].mean() for g in groups]
    frac = [rec_df[rec_df.group == g]["frac_heritable"].mean() for g in groups]
    ax[1].bar(x - w/2, rate, w, color="#bbbbbb", label="tree-only BM rate")
    ax[1].bar(x + w/2, frac, w, color="#2c7fb8", label="decompose frac heritable")
    ax[1].set_xticks(x); ax[1].set_xticklabels(groups)
    ax[1].set_title("(B) Tree-only rate confounds niche;\ndecomposition does not"); ax[1].legend(fontsize=8)
    # (C) identifiability vs intermixing
    ax[2].plot(idf_df.intermixing, idf_df.frac_mae, "o-", color="#2c7fb8")
    ax[2].set_xlabel("clonal intermixing"); ax[2].set_ylabel("mean |recovered - true| frac")
    ax[2].set_title("(C) Identifiability vs intermixing")
    ax[2].invert_xaxis()
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "spatial_decomposition.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    correctness()
    rec = recovery()
    idf = identifiability()
    figure(rec, idf)


if __name__ == "__main__":
    main()
