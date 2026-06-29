"""scPhyTr decomposition vs autocorrelation methods (Hotspot, Moran's I) at separating
heritable from niche-driven genes -- the deconfounding evidence for tl.decompose_variance.

The PEtracer pipeline scores a gene's "heritability" by autocorrelation on the lineage tree and its
"spatial restriction" by autocorrelation in space (Moran's I / Hotspot). But under local tumor
growth spatial proximity ~ phylogenetic proximity, so a niche-driven gene (spatially clustered) is
ALSO tree-autocorrelated and looks heritable, and a heritable gene (clade-clustered) is ALSO
spatially autocorrelated and looks niche-restricted. The two axes are confounded.

We simulate a panel with a KNOWN heritable/niche/mixed split (ph.simulate_spatial_panel) and ask
each method to separate heritable from niche genes:

  * scPhyTr     -- frac_heritable from the additive tree+space decomposition (deconfounded).
  * Moran's I   -- phylogenetic autocorrelation (heritability) and spatial autocorrelation.
  * Hotspot     -- the same, via Hotspot's autocorrelation Z in tree mode and spatial (kNN) mode.

scPhyTr is expected to cleanly separate the classes (high AUROC) where the autocorrelation scores
are confounded (AUROC near chance), and the PEtracer-style tree-vs-space scatter cannot draw a line
between heritable and niche genes.
"""
import os
import numpy as np
import pandas as pd

import scphytr as ph
from analysis.benchmark.spatial_decomposition import _tree, _panel
from analysis.benchmark.path_morans import morans_I, phylo_weights

OUT = os.path.dirname(__file__)


def _auroc(score, label):
    """AUROC that label==1 scores higher than label==0 (rank statistic)."""
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = score[label == 1]; neg = score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    order = np.argsort(score); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[label == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def _leaf_matrix(adata, tree):
    """Per-leaf count matrix Y (n_leaves, n_genes) and log1p-normalized traits, in leaf order."""
    leaves = tree.phylotree.get_leaf_names()
    sp = np.asarray(adata.obs["species"]).astype(str)
    row = {l: np.where(sp == l)[0] for l in leaves}
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    sf = np.asarray(adata.obs["size_factors"], float)
    Y = np.array([X[row[l]].sum(0) for l in leaves])                       # summed counts per leaf
    s = np.array([sf[row[l]].sum() for l in leaves])
    L = np.log1p(Y / np.maximum(s[:, None], 1e-9))                         # log-normalized trait
    return leaves, Y, L


def compute(n=70, n_each=10, intermixing=0.2, spatial_lengthscale=2.5, dispersion=30.0,
            seed=0, use_hotspot=True):
    tree = _tree(n, seed)
    v_ph, v_sp, grp, names = _panel(n_each)
    A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0,
                                  dispersion=[dispersion] * len(names), n_cells=1, mean_size=500,
                                  gene_names=names, seed=seed, intermixing=intermixing,
                                  spatial_lengthscale=spatial_lengthscale)
    ph.pp.setup_anndata(A, tree)
    ph.pp.spatial_neighbors(A, n_neighbors=8)
    ph.tl.decompose_variance(A, dispersion=dispersion)
    A.var["group"] = grp

    leaves, Y, L = _leaf_matrix(A, tree)
    Wt = phylo_weights(tree, leaves); np.fill_diagonal(Wt, 0.0)            # tree weights
    Ws = np.asarray(A.uns["spatial_graph"]["weights"].todense())          # spatial kNN weights
    tree_moran = np.array([morans_I(L[:, g], Wt) for g in range(len(names))])
    space_moran = np.array([morans_I(L[:, g], Ws) for g in range(len(names))])

    df = pd.DataFrame({"group": grp, "frac_heritable": A.var["frac_heritable"].values,
                       "tree_moran": tree_moran, "space_moran": space_moran})

    if use_hotspot:
        try:
            from analysis.kptracer import hotspot_utils as hu
            Lh = L + 1e-3 * np.random.default_rng(seed).standard_normal(L.shape)  # avoid 0-variance
            ht = hu.run_hotspot(Lh, leaves, gene_names=list(names), model="normal", tree=tree.phylotree)
            hs = hu.run_hotspot(Lh, leaves, gene_names=list(names), model="normal",
                                latent=A.uns["spatial_graph"]["coords"])
            df["hotspot_tree_z"] = ht["autocorr"].reindex(names)["Z"].values
            df["hotspot_space_z"] = hs["autocorr"].reindex(names)["Z"].values
        except Exception as e:                                             # hotspot optional
            print(f"[hotspot skipped: {repr(e)[:80]}]")
    return df


def compute_pooled(reps=5, **kw):
    """Pool per-gene results over ``reps`` independent simulations (tree + realizations) for a
    stable AUROC -- single realizations are noisy at the confound boundary."""
    use_hotspot = kw.pop("use_hotspot", True)
    parts = []
    for r in range(reps):
        df = compute(seed=r, use_hotspot=use_hotspot, **kw)
        df["rep"] = r
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def report(df):
    her_ni = df[df.group.isin(["heritable", "niche"])].copy()
    lab = (her_ni.group == "heritable").astype(int).values
    print("\n========== separating HERITABLE from NICHE genes (AUROC) ==========")
    print("  (the 'heritability' read-outs PEtracer uses are tree autocorrelation scores)")
    scores = {"scPhyTr frac_heritable": her_ni.frac_heritable,
              "Moran's I, tree (heritability)": her_ni.tree_moran}
    if "hotspot_tree_z" in df:
        scores["Hotspot, tree autocorr Z"] = her_ni.hotspot_tree_z
    for nm, sc in scores.items():
        print(f"  {nm:34s} AUROC = {_auroc(sc, lab):.2f}")
    print("\n  (a niche gene that scores high on tree-autocorrelation is the confound: "
          "it looks heritable. scPhyTr's frac_heritable is not fooled.)")


def _auroc_row(df):
    hn = df[df.group.isin(["heritable", "niche"])]; lab = (hn.group == "heritable").astype(int).values
    row = {"scPhyTr": _auroc(hn.frac_heritable, lab), "Moran (tree)": _auroc(hn.tree_moran, lab)}
    if "hotspot_tree_z" in df:
        row["Hotspot (tree)"] = _auroc(hn.hotspot_tree_z, lab)
    return row


def sweep(lengthscales=(0.5, 1.0, 1.5, 2.5, 3.5), intermixing=0.2, reps=5, use_hotspot=True):
    """AUROC (heritable vs niche) vs spatial-gradient smoothness -- the confound severity axis."""
    rows = []
    print("\n== AUROC vs spatial smoothness (confound severity), pooled over reps ==")
    for ls in lengthscales:
        df = compute_pooled(reps=reps, intermixing=intermixing, spatial_lengthscale=ls,
                            use_hotspot=use_hotspot)
        r = _auroc_row(df); r["lengthscale"] = ls
        rows.append(r)
        print(f"  lengthscale {ls:.1f}: " + "  ".join(f"{k} {v:.2f}" for k, v in r.items() if k != "lengthscale"))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "spatial_deconfounding_sweep.csv"), index=False)
    return out


def figure(df=None, sw=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None:
        df = pd.read_csv(os.path.join(OUT, "spatial_deconfounding.csv"))
    if sw is None:
        sw = pd.read_csv(os.path.join(OUT, "spatial_deconfounding_sweep.csv"))
    col = {"heritable": "#2c7fb8", "niche": "#e45756", "mixed": "#54a24b"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (A) PEtracer-style naive scatter at the realistic regime -> heritable & niche overlap on tree axis
    for g in col:
        d = df[df.group == g]
        ax[0].scatter(d.tree_moran, d.space_moran, color=col[g], s=40, label=g, edgecolor="k", lw=0.3)
    ax[0].set_xlabel("tree autocorrelation (naive 'heritability')")
    ax[0].set_ylabel("spatial autocorrelation (naive 'niche')")
    ax[0].set_title("(A) Smooth niche gradient + local growth:\nnaive axes confound heritable & niche")
    ax[0].legend(fontsize=8)
    # (B) AUROC vs smoothness -- naive collapses to chance, scPhyTr holds
    method_col = {"scPhyTr": "#2c7fb8", "Moran (tree)": "#d95f0e", "Hotspot (tree)": "#e45756"}
    for m, c in method_col.items():
        if m in sw:
            ax[1].plot(sw.lengthscale, sw[m], "o-", color=c, label=m)
    ax[1].axhline(0.5, ls=":", color="grey")
    ax[1].set_xlabel("spatial gradient smoothness (lengthscale)"); ax[1].set_ylabel("AUROC (heritable vs niche)")
    ax[1].set_ylim(0.4, 1.03); ax[1].set_title("(B) As niche gradients smooth out,\nonly scPhyTr stays accurate")
    ax[1].legend(fontsize=8)
    # (C) scPhyTr frac_heritable separates the classes at the realistic regime
    for i, g in enumerate(["heritable", "mixed", "niche"]):
        d = df[df.group == g]
        ax[2].scatter(np.full(len(d), i) + np.random.uniform(-0.1, 0.1, len(d)), d.frac_heritable,
                      color=col[g], s=40, edgecolor="k", lw=0.3)
    ax[2].set_xticks([0, 1, 2]); ax[2].set_xticklabels(["heritable", "mixed", "niche"])
    ax[2].set_ylabel("scPhyTr frac heritable"); ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_title("(C) scPhyTr frac_heritable\nseparates the classes")
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "spatial_deconfounding.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    df = compute_pooled(reps=5)       # realistic regime: smooth gradient (lengthscale 2.5) + local growth
    df.to_csv(os.path.join(OUT, "spatial_deconfounding.csv"), index=False)
    report(df)
    sw = sweep()
    figure(df, sw)


if __name__ == "__main__":
    main()
