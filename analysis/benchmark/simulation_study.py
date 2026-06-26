"""Parameter-recovery simulation study: does the method recover the TRUE evolutionary rate?

We use scPhyTr's modular simulator (``ph.simulate``: a tree × a BM trait model × a subclonal
Poisson observation) to generate counts with a *known* diffusion rate sigma^2, across a sweep of
sequencing depth, then ask each method to recover sigma^2:

  * scPhyTr  -- the count model through the public API (``ph.tl.estimate_rate``): cells kept as
    subclonal replicates, no pseudobulk.
  * naive    -- the Gaussian-on-log / EvoGeneX-style baseline: collapse each leaf's cells to the
    mean of log1p(count/size) (a pseudobulk), then fit a Gaussian BM rate.

The truth is fixed; the only thing that changes is depth. scPhyTr should track the truth at all
depths, while the naive estimate is biased where shot noise is large (low depth).
"""
import os
import numpy as np
import pandas as pd

import scphytr as ph
from scphytr.trait_models import BrownianMotion
from scphytr.tools.model_selection import fit_bm
from scphytr.utils.tree import Tree
from ete3 import Tree as ETree

OUT = os.path.dirname(__file__)


def _tree(n=60, seed=0):
    rng = np.random.default_rng(seed)
    et = ETree(); et.populate(n, random_branches=True)
    for nd in et.traverse():
        if not nd.is_root():
            nd.dist = max(float(nd.dist), 0.1)
    for i, l in enumerate(et.get_leaves()):
        l.name = f"t{i}"
    tw = Tree(); tw.phylotree = et; tw.root = et.get_tree_root()
    return tw


def _naive_rate(adata, tree, leaves):
    """EvoGeneX-style: per-leaf mean of log1p(count/size) (pseudobulk), Gaussian BM rate."""
    sp = np.asarray(adata.obs["species"]).astype(str)
    s = np.asarray(adata.obs["size_factors"], float)
    out = {}
    for g in adata.var_names:
        y = adata[:, g].X.ravel().astype(float)
        ln = np.log1p(y / np.maximum(s, 1e-9))
        m = pd.Series(ln).groupby(sp).mean()
        vals = {l: float(m.get(l, 0.0)) for l in leaves}
        out[g] = fit_bm(tree, vals).params["sigma2"]
    return out


def main(true_s2=0.7, mu=0.0, n_cells=4, n_genes=60, n_tips=60,
         depths=(1.0, 3.0, 10.0, 50.0, 500.0), seed=0):
    import anndata as adata_mod
    from scphytr.simulation import sample_latent
    tree = _tree(n_tips, seed)
    leaves = tree.phylotree.get_leaf_names(); nL = len(leaves)

    # Fix ONE latent BM realization per gene, shared across all depths, so the comparison
    # isolates sequencing depth (not realization variance).
    latents = []
    for g in range(n_genes):
        tm = BrownianMotion(tree, np.array([mu]), np.array([[true_s2]]))
        z = sample_latent(tree, tm, np.random.default_rng(1000 + g))
        latents.append(np.array([z[l] for l in tree.root.get_leaves()]))
    L = np.column_stack(latents)                                          # (nL, n_genes)

    rows = []
    for d in depths:
        rng = np.random.default_rng(7000 + int(d))
        idx = np.repeat(np.arange(nL), n_cells)
        sizes = rng.gamma(4.0, d / 4.0, idx.shape[0]) / d                 # per-cell size (mean 1)
        X = rng.poisson((sizes * d)[:, None] * np.exp(L[idx])).astype(float)
        adata = adata_mod.AnnData(X=X)
        adata.var_names = [f"gene{g}" for g in range(n_genes)]
        adata.obs["species"] = [leaves[i] for i in idx]
        adata.obs["size_factors"] = sizes
        ph.pp.setup_anndata(adata, tree)
        ph.tl.estimate_rate(adata)                       # scPhyTr (count model, no pseudobulk)
        sp = adata.var["rate"].values
        nv = np.array(list(_naive_rate(adata, tree, leaves).values()))   # Gaussian-on-log pseudobulk
        rows.append({"depth": d, "scphytr": float(np.median(sp)), "naive": float(np.median(nv)),
                     "scphytr_mean": float(np.mean(sp)), "naive_mean": float(np.mean(nv))})
        print(f"depth {d:6.0f}: scPhyTr median σ²={np.median(sp):.2f}  "
              f"naive median σ²={np.median(nv):.2f}  (true {true_s2})")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "simulation_study.csv"), index=False)

    print("\n========== rate recovery (true σ² = %.2f) ==========" % true_s2)
    sp_err = np.abs(df["scphytr"] / true_s2 - 1).mean()
    nv_err = np.abs(df["naive"] / true_s2 - 1).mean()
    print(f"mean |relative bias| across depths: scPhyTr {sp_err:.0%}, naive {nv_err:.0%}")
    _figure(df, true_s2)
    return df


def _figure(df, true_s2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.axhline(true_s2, ls="--", color="k", lw=1, label=f"true σ² = {true_s2}")
    ax.plot(df["depth"], df["scphytr"], "o-", color="#2c7fb8", label="scPhyTr (count model)")
    ax.plot(df["depth"], df["naive"], "s-", color="#d95f0e", label="naive Gaussian-on-log (pseudobulk)")
    ax.set_xscale("log"); ax.set_xlabel("sequencing depth (mean counts/cell)")
    ax.set_ylabel("estimated diffusion rate $\\hat\\sigma^2$ (median over genes)")
    ax.set_title("Recovering the true evolutionary rate vs depth"); ax.legend(fontsize=9)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "simulation_study.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
