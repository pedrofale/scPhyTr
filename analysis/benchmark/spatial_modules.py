"""Deconfounded spatial gene programs: scPhyTr vs Hotspot at recovering niche modules.

PEtracer/Yosef use Hotspot to find spatially-coherent gene modules. But under local growth a clonal
(heritable) gene is spatially clustered too, so Hotspot's spatial autocorrelation groups clonal
genes into spurious "spatial" modules and links them to real niche programs. scPhyTr first removes
the lineage component (the additive tree+space decomposition) and reads the gene-gene correlation of
the *spatial* components, so only genuine co-niche genes group together.

We plant two spatial modules (each a shared niche field) plus heritable genes (clonal, no niche),
and score each method's gene-gene association matrix by how well it isolates the true same-module
pairs from everything else (cross-module, clonal-clonal, clonal-niche):

  * scPhyTr     -- uns['niche_corr'] from tl.spatial_programs (lineage-removed spatial components).
  * Hotspot     -- spatial (kNN) local-correlation Z on raw expression.
  * naive       -- Pearson correlation of per-leaf log expression.
"""
import os
import numpy as np
import pandas as pd

import scphytr as ph
from analysis.benchmark.spatial_decomposition import _tree

OUT = os.path.dirname(__file__)


def _auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[label == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def _panel(nA=5, nB=5, nC=6):
    """Two spatial modules A,B (shared niche field) + a CO-HERITABLE module C (shared clonal field,
    no shared niche) -- the confounder that local growth makes look spatial."""
    groups = np.array(["A"] * nA + ["B"] * nB + ["C"] * nC)
    p = nA + nB + nC
    v_sp = np.r_[np.full(nA, 2.0), np.full(nB, 2.0), np.full(nC, 0.05)]
    v_ph = np.r_[np.full(nA, 0.1), np.full(nB, 0.1), np.full(nC, 2.0)]
    spatial_module = np.r_[np.full(nA, 0), np.full(nB, 1), np.arange(90, 90 + nC)]   # C: no shared niche
    phylo_module = np.r_[np.arange(200, 200 + nA + nB), np.full(nC, 100)]            # C: shared BM
    names = [f"A{i}" for i in range(nA)] + [f"B{i}" for i in range(nB)] + [f"C{i}" for i in range(nC)]
    return v_ph, v_sp, spatial_module, phylo_module, groups, names


def _pairs(groups):
    """Per gene-pair: same_spatial_module (true co-niche) and co_heritable (confounder, not co-niche)."""
    p = len(groups); iu = np.triu_indices(p, 1)
    gi, gj = groups[iu[0]], groups[iu[1]]
    same = ((gi == gj) & np.isin(gi, ["A", "B"])).astype(int)
    coher = ((gi == "C") & (gj == "C")).astype(int)
    return iu, same, coher


def compute(n=70, intermixing=0.0, spatial_lengthscale=1.0, dispersion=30.0, growth="clonal", seed=0):
    tree = _tree(n, seed)
    v_ph, v_sp, spatial_module, phylo_module, groups, names = _panel()
    A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0,
                                  dispersion=[dispersion] * len(names), n_cells=1, mean_size=500,
                                  spatial_lengthscale=spatial_lengthscale, intermixing=intermixing,
                                  growth=growth, spatial_module=spatial_module,
                                  phylo_module=phylo_module, gene_names=names, seed=seed)
    ph.pp.setup_anndata(A, tree); ph.pp.spatial_neighbors(A, n_neighbors=8)
    ph.tl.spatial_programs(A, dispersion=dispersion)
    K_sc = np.abs(A.uns["niche_corr"])

    # per-leaf log-expression matrix (leaf order) for Hotspot / naive
    leaves = tree.phylotree.get_leaf_names()
    sp = np.asarray(A.obs["species"]).astype(str)
    X = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
    sf = np.asarray(A.obs["size_factors"], float)
    row = {l: np.where(sp == l)[0] for l in leaves}
    Y = np.array([X[row[l]].sum(0) for l in leaves]); s = np.array([sf[row[l]].sum() for l in leaves])
    L = np.log1p(Y / np.maximum(s[:, None], 1e-9))
    K_naive = np.abs(np.nan_to_num(np.corrcoef(L, rowvar=False)))

    K_hs = None
    try:
        from analysis.kptracer import hotspot_utils as hu
        Lh = L + 1e-3 * np.random.default_rng(seed).standard_normal(L.shape)
        res = hu.run_hotspot(Lh, leaves, gene_names=list(names), model="normal",
                             latent=A.uns["spatial_graph"]["coords"], restrict_genes=list(names))
        if res["lcz"] is not None:
            K_hs = np.abs(np.nan_to_num(res["lcz"].reindex(index=names, columns=names).values))
    except Exception as e:
        print(f"[hotspot skipped: {repr(e)[:80]}]")

    iu, same, coher = _pairs(groups)
    out = {"scPhyTr": K_sc[iu], "naive": K_naive[iu]}
    if K_hs is not None:
        out["Hotspot"] = K_hs[iu]
    return pd.DataFrame({**out, "same_module": same, "coheritable": coher}), groups, names


def _spurious_rate(df, m):
    """Fraction of co-heritable (module C) pairs that out-score the median TRUE co-niche pair --
    i.e. a co-heritable clonal program falsely called a spatial module."""
    thr = df.loc[df.same_module == 1, m].median()
    cc = df[df.coheritable == 1]
    return float((cc[m] > thr).mean()) if len(cc) else np.nan


def report(df):
    print("\n========== deconfounded spatial modules: true co-niche vs co-heritable confounder ==========")
    methods = [m for m in ["scPhyTr", "Hotspot", "naive"] if m in df]
    for m in methods:
        au = _auroc(df[m], df.same_module)
        print(f"  {m:9s} AUROC(true co-niche) = {au:.2f}   "
              f"| co-heritable pairs falsely called co-niche = {_spurious_rate(df, m):.0%}")
    print("\n  (under local growth a co-heritable clonal program is spatially clustered. NAIVE "
          "correlation calls it a spatial module; scPhyTr -- by explicitly removing the lineage "
          "component -- and Hotspot's spatial-locality null both largely resist it.)")


def compute_pooled(reps=5, **kw):
    parts = []
    for r in range(reps):
        df, groups, names = compute(seed=r, **kw)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def figure(df=None, example=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None:
        df = pd.read_csv(os.path.join(OUT, "spatial_modules.csv"))
    methods = [m for m in ["scPhyTr", "Hotspot", "naive"] if m in df]
    fig, ax = plt.subplots(1, 2 + (example is not None), figsize=(5 * (2 + (example is not None)), 4.4))
    # (A) AUROC (true co-niche) + co-heritable false-module rate
    x = np.arange(len(methods)); w = 0.38
    au = [_auroc(df[m], df.same_module) for m in methods]
    fp = [_spurious_rate(df, m) for m in methods]
    ax[0].bar(x - w/2, au, w, color="#2c7fb8", label="AUROC (true co-niche)")
    ax[0].bar(x + w/2, fp, w, color="#e45756", label="co-heritable called co-niche")
    ax[0].axhline(0.5, ls=":", color="grey")
    ax[0].set_xticks(x); ax[0].set_xticklabels(methods); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(A) Module recovery & clonal confound"); ax[0].legend(fontsize=8)
    # (B) association distributions by pair type, scPhyTr
    cats = [("true co-niche", df.same_module == 1, "#2c7fb8"),
            ("cross/other", (df.same_module == 0) & (df.coheritable == 0), "#999999"),
            ("co-heritable", df.coheritable == 1, "#e45756")]
    for i, (lab, mask, c) in enumerate(cats):
        ax[1].scatter(np.full(mask.sum(), i) + np.random.uniform(-0.12, 0.12, mask.sum()),
                      df.loc[mask, "scPhyTr"], s=8, color=c, alpha=0.5)
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels([c[0] for c in cats], fontsize=8)
    ax[1].set_ylabel("scPhyTr niche |corr|"); ax[1].set_title("(B) scPhyTr: only true co-niche pairs score high")
    # (C) optional example niche-corr heatmap
    if example is not None:
        K, names = example
        im = ax[2].imshow(np.abs(K), cmap="magma", vmin=0, vmax=1)
        ax[2].set_title("(C) scPhyTr niche correlation\n(2 modules + heritable)")
        ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=90, fontsize=6)
        ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=6)
        fig.colorbar(im, ax=ax[2], shrink=0.7)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "spatial_modules.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    df = compute_pooled(reps=5)
    df.to_csv(os.path.join(OUT, "spatial_modules.csv"), index=False)
    report(df)
    # rebuild one example niche_corr for the heatmap
    tree = _tree(70, 0); v_ph, v_sp, sm, pm, groups, names = _panel()
    A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0, dispersion=[30.] * len(names),
                                  n_cells=1, mean_size=500, spatial_lengthscale=1.0, intermixing=0.0,
                                  growth="clonal", spatial_module=sm, phylo_module=pm,
                                  gene_names=names, seed=0)
    ph.pp.setup_anndata(A, tree); ph.tl.spatial_programs(A, dispersion=30.0)
    figure(df, example=(A.uns["niche_corr"], names))


if __name__ == "__main__":
    main()
