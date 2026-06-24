"""Real KP-Tracer data: are the Hotspot gene modules tree-confounded?

We reproduce the paper's module step -- Hotspot on the scVI-latent KNN graph
(their Figure 3 setup) -- per tumor, and also run Hotspot in tree mode
(PhyloVision style). For every module Hotspot reports we ask whether its
gene-gene coherence survives deconfounding the phylogeny:

  * naive within-module correlation   : mean |Pearson r| of module gene pairs at
    the leaves (the co-expression Hotspot responds to);
  * deconfounded within-module corr   : the same on Felsenstein contrasts (whiten
    by the tumor's phylogenetic covariance C) -- the off-diagonal of the
    evolutionary rate matrix K, scPhyTr's target;
  * module clade variance fraction eta^2: how much of the module score's variance
    sits BETWEEN deep clones (clonal identity / phylogenetic signal).

A genuine co-regulation module keeps its correlation after deconfounding; a
tree-confounded module -- genes that merely co-drift down shared branches -- has
high naive correlation that shrinks on contrasts and high clade eta^2. Pooling
modules across tumors, the prediction is a positive shrinkage-vs-eta^2 trend:
the more clonal a Hotspot module, the more of its "co-expression" is the tree.

Outputs analysis/kptracer/figures/hotspot_real.png and prints numbers.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".numba_cache"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from analysis.kptracer.load import load_tumor
from analysis.kptracer.phylo_factor_utils import get_clades, clade_eta2
from analysis.kptracer import hotspot_utils as hu

DATA = "data/external/KPTracer-Data"
HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")

MODES = [("knn", "knn (scVI, their setup)", "#fdae6b", "o"),
         ("tree", "tree (PhyloVision)", "#d95f0e", "s")]


def _within_module_pairs(module_genes, gene_idx, Rt, Rc):
    idx = [gene_idx[g] for g in module_genes if g in gene_idx]
    tip, con = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            tip.append(abs(Rt[idx[a], idx[b]]))
            con.append(abs(Rc[idx[a], idx[b]]))
    return np.array(tip), np.array(con)


def run_one(tumor, n_hvg=200, Q=10, adata=None):
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=n_hvg)
    C, Y, genes, names, tree = d["C"], d["Y"], d["genes"], d["leaf_names"], d["tree"]
    n, p = Y.shape
    gene_idx = {g: i for i, g in enumerate(genes)}
    labels = get_clades(tree, names, Q)
    Ystd = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    Rt, _ = hu.tip_corr(Y)
    Rc, _ = hu.contrast_corr(Y, C)

    print(f"\n=== {tumor} ({n} cells, {p} HVGs) ===")
    out = {"tumor": tumor, "n": n}
    for key, label, _, _ in MODES:
        kw = dict(latent=np.asarray(adata[names].obsm["X_scVI"], float)) if key == "knn" \
            else dict(tree=tree)
        res = hu.run_hotspot(Y, names, gene_names=genes, model="normal",
                             restrict_genes=None, fdr=0.05, jobs=1, **kw)
        try:
            mods = res["hs"].create_modules(min_gene_threshold=max(10, p // 20),
                                            core_only=False, fdr_threshold=0.05)
        except Exception as e:
            print(f"  [{label}] create_modules failed: {e}")
            out[key] = []
            continue
        recs = []
        for m in [m for m in sorted(mods.unique()) if m != -1]:
            mg = list(mods.index[mods == m])
            tip, con = _within_module_pairs(mg, gene_idx, Rt, Rc)
            if len(tip) == 0:
                continue
            mi = [gene_idx[g] for g in mg if g in gene_idx]
            eta = clade_eta2(Ystd[:, mi].mean(1), labels)
            shrink = 1 - con.mean() / (tip.mean() + 1e-12)
            recs.append(dict(tumor=tumor, m=int(m), ngenes=len(mi), tip=tip.mean(),
                             con=con.mean(), shrink=shrink, eta=eta,
                             tip_pairs=tip, con_pairs=con))
            print(f"  [{label}] module {m}: {len(mi):3d} genes | |r| naive "
                  f"{tip.mean():.2f} -> deconf {con.mean():.2f} ({100*shrink:+.0f}%) "
                  f"| eta^2 {eta:.2f}")
        out[key] = recs
    return out


def run(tumors=None, focal=None, n_hvg=200, Q=10, adata=None):
    assert adata is not None, "pass a preloaded AnnData (needs obsm['X_scVI'])"
    if tumors is None:
        tumors = ["3432_NT_T1", "3435_NT_T3", "3513_NT_T5", "3430_NT_T2",
                  "3730_NT_T2", "3726_NT_T2", "3513_NT_T1", "3434_NT_T2",
                  "3513_NT_T3", "3430_NT_T1"]
    focal = focal or "3730_NT_T2"
    all_runs = [run_one(t, n_hvg=n_hvg, Q=Q, adata=adata) for t in tumors]
    focal_run = next(r for r in all_runs if r["tumor"] == focal)

    # pooled per-module records by mode
    pooled = {key: [r for run_ in all_runs for r in run_.get(key, [])]
              for key, *_ in MODES}
    for key, label, *_ in MODES:
        recs = pooled[key]
        if not recs:
            continue
        eta = np.array([r["eta"] for r in recs]); shr = np.array([r["shrink"] for r in recs])
        rho = stats.spearmanr(eta, shr).correlation if len(recs) > 2 else float("nan")
        print(f"\n[pooled {label}] {len(recs)} modules across {len(tumors)} tumors | "
              f"Spearman(eta^2, shrinkage) = {rho:.2f} | "
              f"clonal modules (eta^2>0.2): mean shrink "
              f"{100*np.mean([r['shrink'] for r in recs if r['eta']>0.2] or [0]):.0f}%, "
              f"state modules (eta^2<0.1): mean shrink "
              f"{100*np.mean([r['shrink'] for r in recs if r['eta']<0.1] or [0]):.0f}%")

    _figure(focal_run, pooled, len(tumors))
    return dict(runs=all_runs, pooled=pooled)


def _figure(focal_run, pooled, n_tumors):
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (A) within-module pair correlations on the focal tumor (their scVI-KNN modules)
    recs = focal_run.get("knn", []) or focal_run.get("tree", [])
    if recs:
        tip = np.concatenate([r["tip_pairs"] for r in recs])
        con = np.concatenate([r["con_pairs"] for r in recs])
        ax[0].scatter(tip, con, s=10, alpha=0.3, color="#fdae6b", edgecolor="none")
        lim = max(tip.max(), con.max(), 0.3) * 1.05
        ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="no deconfounding")
        ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
        ax[0].set_xlabel("naive |corr| at leaves (Hotspot sees this)")
        ax[0].set_ylabel("deconfounded |corr| (contrasts)")
        ax[0].legend(fontsize=8, loc="upper left")
        ax[0].set_title(f"(A) {focal_run['tumor']}: within scVI-KNN\nHotspot modules, "
                        "pairs sit below the line", fontsize=10)

    # (B) per-module naive vs deconfounded mean |r| on the focal tumor
    x = 0; ticks, labs = [], []
    for key, label, col, _ in MODES:
        for r in focal_run.get(key, []):
            ax[1].plot([x, x], [r["con"], r["tip"]], color="grey", lw=1, zorder=1)
            ax[1].scatter([x], [r["tip"]], s=40, color=col, zorder=3,
                          label="naive" if not ticks else None)
            ax[1].scatter([x], [r["con"]], s=40, color="#2c7fb8", zorder=3,
                          label="deconfounded" if not ticks else None)
            ticks.append(x); labs.append(f"{key[0].upper()}{r['m']}"); x += 1
        x += 1
    ax[1].set_xticks(ticks); ax[1].set_xticklabels(labs, fontsize=7)
    ax[1].set_ylabel("mean within-module |corr|")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"(B) {focal_run['tumor']}: module coherence\nvs deconfounded", fontsize=10)

    # (C) pooled shrinkage vs clonal signal across tumors
    for key, label, col, mk in MODES:
        recs = pooled.get(key, [])
        if not recs:
            continue
        eta = [r["eta"] for r in recs]; shr = [100 * r["shrink"] for r in recs]
        ax[2].scatter(eta, shr, s=45, color=col, marker=mk, edgecolor="k",
                      alpha=0.85, label=f"{label.split()[0]} ({len(recs)})")
    ax[2].axhline(0, ls=":", color="grey")
    ax[2].set_xlabel(r"module clade variance fraction $\eta^2$ (clonal signal)")
    ax[2].set_ylabel("% within-module corr lost\non deconfounding the tree")
    ax[2].legend(fontsize=8, title=f"{n_tumors} tumors")
    ax[2].set_title("(C) The more clonal a Hotspot module,\nthe more it is a tree artifact",
                    fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "hotspot_real.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[fig] wrote {out}")


if __name__ == "__main__":
    import anndata
    ad = anndata.read_h5ad(DATA + "/expression/adata_processed.nt.h5ad")
    run(adata=ad)
