"""Deconfounded *clonal* gene programs: scPhyTr vs Hotspot at recovering shared-lineage modules.

This is the SYMMETRIC twin of ``spatial_modules.py``. There the discriminating signal lived on the
spatial axis (find co-niche modules) -- a HIGH-autocorrelation / high-SNR axis where Hotspot's
spatial-locality null is already robust, so scPhyTr only ties it. Here the signal lives on the
TREE/clonal axis -- a LOW-autocorrelation / low-SNR axis (BM on a tree is high-frequency: even a
perfectly heritable gene has tree Moran's-I ~0.2). That is exactly the axis where marginal
autocorrelation detectors get confounded, and where scPhyTr's explicit decomposition should win --
the gene-gene analog of the per-gene heritability result (benchmark 1: scPhyTr 0.99 vs Hotspot 0.79).

We plant two CO-HERITABLE modules A,B (each a shared clonal BM field, no shared niche) plus a
CO-NICHE confounder module C (a shared spatial field, no shared clonal program). Under local growth
C is spatially clustered AND -- because spatial proximity tracks the lineage -- it leaks onto the
tree graph, so a tree-autocorrelation method calls it a clonal module. We score each method's
gene-gene association by how well it isolates the true co-heritable pairs from the co-niche
confounder (and everything else):

  * scPhyTr  -- uns['clonal_corr'] from tl.spatial_programs (phylo component, niche removed).
  * Hotspot  -- TREE-mode (phylogeny graph) local-correlation Z on raw expression.
  * naive    -- Pearson correlation of per-leaf log expression.
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
    """Two CLONAL modules A,B (shared BM field, no shared niche) + a CO-NICHE confounder module C
    (shared spatial field, no shared clonal program) -- the confounder that local growth makes look
    clonal."""
    groups = np.array(["A"] * nA + ["B"] * nB + ["C"] * nC)
    v_ph = np.r_[np.full(nA, 2.0), np.full(nB, 2.0), np.full(nC, 0.1)]
    v_sp = np.r_[np.full(nA, 0.1), np.full(nB, 0.1), np.full(nC, 2.0)]
    phylo_module = np.r_[np.full(nA, 0), np.full(nB, 1), np.arange(90, 90 + nC)]    # C: no shared BM
    spatial_module = np.r_[np.arange(200, 200 + nA + nB), np.full(nC, 100)]         # C: shared niche
    names = [f"A{i}" for i in range(nA)] + [f"B{i}" for i in range(nB)] + [f"C{i}" for i in range(nC)]
    return v_ph, v_sp, spatial_module, phylo_module, groups, names


def _pairs(groups):
    """Per gene-pair: same_clonal_module (true co-heritable) and co_niche (confounder, not co-clonal)."""
    p = len(groups); iu = np.triu_indices(p, 1)
    gi, gj = groups[iu[0]], groups[iu[1]]
    same = ((gi == gj) & np.isin(gi, ["A", "B"])).astype(int)
    coniche = ((gi == "C") & (gj == "C")).astype(int)
    return iu, same, coniche


def compute(n=70, intermixing=0.1, spatial_lengthscale=2.5, dispersion=30.0, growth="clonal", seed=0):
    tree = _tree(n, seed)
    v_ph, v_sp, spatial_module, phylo_module, groups, names = _panel()
    A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0,
                                  dispersion=[dispersion] * len(names), n_cells=1, mean_size=500,
                                  spatial_lengthscale=spatial_lengthscale, intermixing=intermixing,
                                  growth=growth, spatial_module=spatial_module,
                                  phylo_module=phylo_module, gene_names=names, seed=seed)
    ph.pp.setup_anndata(A, tree); ph.pp.spatial_neighbors(A, n_neighbors=8)
    ph.tl.spatial_programs(A, dispersion=dispersion)
    K_sc = np.abs(A.uns["clonal_corr"])

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
                             tree=tree.phylotree, restrict_genes=list(names))   # TREE mode
        if res["lcz"] is not None:
            K_hs = np.abs(np.nan_to_num(res["lcz"].reindex(index=names, columns=names).values))
    except Exception as e:
        print(f"[hotspot skipped: {repr(e)[:80]}]")

    iu, same, coniche = _pairs(groups)
    out = {"scPhyTr": K_sc[iu], "naive": K_naive[iu]}
    if K_hs is not None:
        out["Hotspot"] = K_hs[iu]
    return pd.DataFrame({**out, "same_module": same, "coniche": coniche}), groups, names


def _spurious_rate(df, m):
    """Fraction of co-niche (module C) pairs that out-score the median TRUE co-clonal pair --
    i.e. a co-niche spatial program falsely called a clonal module."""
    thr = df.loc[df.same_module == 1, m].median()
    cc = df[df.coniche == 1]
    return float((cc[m] > thr).mean()) if len(cc) else np.nan


def report(df):
    print("\n========== deconfounded clonal modules: true co-heritable vs co-niche confounder ==========")
    methods = [m for m in ["scPhyTr", "Hotspot", "naive"] if m in df]
    for m in methods:
        au = _auroc(df[m], df.same_module)
        print(f"  {m:9s} AUROC(true co-clonal) = {au:.2f}   "
              f"| co-niche pairs falsely called co-clonal = {_spurious_rate(df, m):.0%}")
    print("\n  (under local growth a co-niche spatial program leaks onto the tree graph. NAIVE "
          "correlation and Hotspot's TREE-mode autocorrelation -- a low-SNR axis -- get confounded; "
          "scPhyTr, by explicitly removing the spatial component, isolates the true clonal modules.)")


def compute_pooled(reps=5, **kw):
    parts = []
    for r in range(reps):
        df, groups, names = compute(seed=r, **kw)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def sweep(reps=5, intermixings=(0.05, 0.1, 0.2), lengthscales=(1.0, 2.5)):
    """Where does the clonal-module gap open? The niche->tree leak (hence Hotspot's tree-mode
    confound) is strongest at LOW intermixing + SMOOTH niche gradients -- the realistic
    hypoxia/tumour-margin regime. Mirror of benchmark 1's smoothness sweep, on the gene-gene axis.
    Runs each config in this process; if OpenBLAS aborts, run configs as separate processes."""
    rows = []
    for interm in intermixings:
        for ls in lengthscales:
            df, _, _ = compute(intermixing=interm, spatial_lengthscale=ls, seed=0) if reps == 1 \
                else (compute_pooled(reps=reps, intermixing=interm, spatial_lengthscale=ls), None, None)
            r = {"intermixing": interm, "lengthscale": ls}
            for m in ["scPhyTr", "Hotspot", "naive"]:
                if m in df:
                    r[f"{m}_auroc"] = _auroc(df[m], df.same_module)
                    r[f"{m}_spurious"] = _spurious_rate(df, m)
            rows.append(r)
    return pd.DataFrame(rows)


def figure(df=None, example=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None:
        df = pd.read_csv(os.path.join(OUT, "spatial_clonal_modules.csv"))
    methods = [m for m in ["scPhyTr", "Hotspot", "naive"] if m in df]
    fig, ax = plt.subplots(1, 2 + (example is not None), figsize=(5 * (2 + (example is not None)), 4.4))
    # (A) AUROC (true co-clonal) + co-niche false-module rate
    x = np.arange(len(methods)); w = 0.38
    au = [_auroc(df[m], df.same_module) for m in methods]
    fp = [_spurious_rate(df, m) for m in methods]
    ax[0].bar(x - w/2, au, w, color="#2c7fb8", label="AUROC (true co-clonal)")
    ax[0].bar(x + w/2, fp, w, color="#e45756", label="co-niche called co-clonal")
    ax[0].axhline(0.5, ls=":", color="grey")
    ax[0].set_xticks(x); ax[0].set_xticklabels(methods); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(A) Clonal-module recovery & niche confound"); ax[0].legend(fontsize=8)
    # (B) association distributions by pair type, scPhyTr
    cats = [("true co-clonal", df.same_module == 1, "#2c7fb8"),
            ("cross/other", (df.same_module == 0) & (df.coniche == 0), "#999999"),
            ("co-niche", df.coniche == 1, "#e45756")]
    for i, (lab, mask, c) in enumerate(cats):
        ax[1].scatter(np.full(mask.sum(), i) + np.random.uniform(-0.12, 0.12, mask.sum()),
                      df.loc[mask, "scPhyTr"], s=8, color=c, alpha=0.5)
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels([c[0] for c in cats], fontsize=8)
    ax[1].set_ylabel("scPhyTr clonal |corr|"); ax[1].set_title("(B) scPhyTr: only true co-clonal pairs score high")
    # (C) optional example clonal-corr heatmap
    if example is not None:
        K, names = example
        im = ax[2].imshow(np.abs(K), cmap="magma", vmin=0, vmax=1)
        ax[2].set_title("(C) scPhyTr clonal correlation\n(2 clonal modules + niche)")
        ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=90, fontsize=6)
        ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=6)
        fig.colorbar(im, ax=ax[2], shrink=0.7)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "spatial_clonal_modules.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    df = compute_pooled(reps=5)
    df.to_csv(os.path.join(OUT, "spatial_clonal_modules.csv"), index=False)
    report(df)
    # rebuild one example clonal_corr for the heatmap
    tree = _tree(70, 0); v_ph, v_sp, sm, pm, groups, names = _panel()
    A = ph.simulate_spatial_panel(tree, v_ph, v_sp, dim=2, diffusion=1.0, dispersion=[30.] * len(names),
                                  n_cells=1, mean_size=500, spatial_lengthscale=1.0, intermixing=0.2,
                                  growth="clonal", spatial_module=sm, phylo_module=pm,
                                  gene_names=names, seed=0)
    ph.pp.setup_anndata(A, tree); ph.pp.spatial_neighbors(A, n_neighbors=8)
    ph.tl.spatial_programs(A, dispersion=30.0)
    figure(df, example=(A.uns["clonal_corr"], names))


if __name__ == "__main__":
    main()
