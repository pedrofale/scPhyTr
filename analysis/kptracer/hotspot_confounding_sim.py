"""Ground-truth control: are Hotspot 'modules' confounded by the phylogeny?

We keep a REAL KP-Tracer tumor tree but simulate expression so we know the truth:
every gene is an *independent* Brownian motion on the tree (K diagonal -> there is
NO real gene module, no real co-regulation). Any gene-gene association a method
reports is therefore a tree artifact.

We then build the same gene x gene association Z-matrix four ways and BH-correct
identically (analysis/kptracer/hotspot_utils.py):

  tip            naive Pearson correlation at the leaves
  hotspot-knn    Hotspot on a KNN graph in a latent space (their Figure-3 setup,
                 latent = PCA of the simulated expression, which inherits lineage)
  hotspot-tree   Hotspot with the phylogeny as the cell graph (PhyloVision mode)
  phylo          Felsenstein contrasts -> deconfounded correlation (scPhyTr's target)

Because the truth is "no module", the number of significant gene pairs should be
~0. tip / hotspot over-call (cell-exchangeability null is violated by shared
ancestry); the contrast estimator stays calibrated. Over many histories we report
the false-positive rate P(>=1 significant pair).

Outputs analysis/kptracer/figures/hotspot_confounding_<tumor>.png and prints numbers.
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
from analysis.kptracer.phylo_factor_utils import chol, simulate_independent_genes
from analysis.kptracer import hotspot_utils as hu

DATA = "data/external/KPTracer-Data"
HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")

COLORS = {"tip": "#969696", "knn": "#fdae6b", "tree": "#d95f0e", "phylo": "#2c7fb8"}
LABELS = {"tip": "naive\ntip corr", "knn": "Hotspot\n(KNN)",
          "tree": "Hotspot\n(tree)", "phylo": "phylo\n(contrasts)"}


def _pca_latent(Y, d=10):
    Yc = Y - Y.mean(0)
    U, s, _ = np.linalg.svd(Yc, full_matrices=False)
    d = min(d, U.shape[1])
    return U[:, :d] * s[:d]


def run(tumor="3726_NT_T2", p=30, R=30, adata=None, seed=0, jobs=1):
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=1)   # expression unused
    C, names, tree = d["C"], d["leaf_names"], d["tree"]
    n = len(names)
    L_C = chol(C)
    print(f"\n=== Hotspot confounding control on real tree {tumor} ({n} leaves) ===")
    print(f"genes EVOLUTIONARILY INDEPENDENT on the tree (K diagonal): no true module")

    # ---------- one representative history ----------------------------------
    rng = np.random.default_rng(seed)
    Y = simulate_independent_genes(L_C, p, rng)
    n_pairs = p * (p - 1) // 2

    # naive tip + phylo contrasts
    Rt, mt = hu.tip_corr(Y)
    z_tip, _ = hu.upper_tri_z(hu.corr_to_z(Rt, mt))
    Rc, mc = hu.contrast_corr(Y, C)
    z_phy, _ = hu.upper_tri_z(hu.corr_to_z(Rc, mc))

    # Hotspot tree + knn (force all p genes so the comparison is the full p x p)
    gnames = [f"g{j}" for j in range(p)]
    res_tree = hu.run_hotspot(Y, names, model="normal", tree=tree,
                              gene_names=gnames, restrict_genes=gnames, jobs=jobs)
    res_knn = hu.run_hotspot(Y, names, model="normal", latent=_pca_latent(Y),
                             gene_names=gnames, restrict_genes=gnames, jobs=jobs)
    z_tree, _ = hu.upper_tri_z(res_tree["lcz"])
    z_knn, _ = hu.upper_tri_z(res_knn["lcz"])

    zvals = {"tip": z_tip, "knn": z_knn, "tree": z_tree, "phylo": z_phy}
    nsig = {k: hu.n_significant_pairs(v)[0] for k, v in zvals.items()}
    ac_tree_sig = int((res_tree["autocorr"].FDR < 0.05).sum())

    # Hotspot's own module call (their language): how many genes get put in a module?
    n_mod, n_in_mod = 0, 0
    try:
        mods = res_tree["hs"].create_modules(min_gene_threshold=max(5, p // 6),
                                             core_only=False, fdr_threshold=0.05)
        n_mod = int((mods.unique() != -1).sum())
        n_in_mod = int((mods != -1).sum())
    except Exception as e:
        print("  (create_modules:", e, ")")

    print(f"\n[representative history] {n_pairs} gene pairs, truth = 0 associated")
    print(f"  Hotspot tree-mode flags {ac_tree_sig}/{p} genes 'significantly "
          f"autocorrelated' (every drifting gene looks heritable)")
    print(f"  Hotspot tree-mode local_corr_z off-diag: sd={z_tree.std():.2f} "
          f"(nominal 1.0); |z|>1.96 in {100*np.mean(np.abs(z_tree)>1.96):.0f}% of pairs")
    print(f"  phylo contrast-corr z off-diag:          sd={z_phy.std():.2f} (calibrated)")
    print(f"  # significant pairs after BH-FDR: " +
          ", ".join(f"{LABELS[k].splitlines()[0]}={nsig[k]}" for k in ["tip", "knn", "tree", "phylo"]))
    print(f"  Hotspot tree-mode 'create_modules': {n_mod} module(s), "
          f"{n_in_mod}/{p} genes assigned (truth: none)")

    # ---------- repeated histories: false-positive rate ---------------------
    rng2 = np.random.default_rng(100)
    keys = ["tip", "knn", "tree", "phylo"]
    any_sig = {k: np.zeros(R, bool) for k in keys}
    mean_sig = {k: np.zeros(R) for k in keys}
    for r in range(R):
        Yr = simulate_independent_genes(L_C, p, rng2)
        Rt, mt = hu.tip_corr(Yr); zt, _ = hu.upper_tri_z(hu.corr_to_z(Rt, mt))
        Rc, mc = hu.contrast_corr(Yr, C); zp, _ = hu.upper_tri_z(hu.corr_to_z(Rc, mc))
        gnames = [f"g{j}" for j in range(p)]
        rt = hu.run_hotspot(Yr, names, model="normal", tree=tree,
                            gene_names=gnames, restrict_genes=gnames, jobs=jobs)
        rk = hu.run_hotspot(Yr, names, model="normal", latent=_pca_latent(Yr),
                            gene_names=gnames, restrict_genes=gnames, jobs=jobs)
        zk, _ = hu.upper_tri_z(rk["lcz"]); ztr, _ = hu.upper_tri_z(rt["lcz"])
        for k, zz in [("tip", zt), ("knn", zk), ("tree", ztr), ("phylo", zp)]:
            s = hu.n_significant_pairs(zz)[0]
            mean_sig[k][r] = s
            any_sig[k][r] = s > 0
        if (r + 1) % 10 == 0:
            print(f"  ...history {r+1}/{R}")

    rate = {k: float(any_sig[k].mean()) for k in keys}
    print(f"\n[over {R} histories] P(>=1 'significant' gene pair), nominal 0.05:")
    for k in keys:
        print(f"  {LABELS[k].replace(chr(10),' '):20s}: P={rate[k]:.2f}  "
              f"(mean # sig pairs {mean_sig[k].mean():.1f} of {n_pairs})")

    # ---------- figure ------------------------------------------------------
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    grid = np.linspace(-6, 6, 100)
    ax[0].hist(z_tree, bins=np.linspace(-8, 8, 40), density=True, color=COLORS["tree"],
               alpha=0.6, label=f"Hotspot tree (sd {z_tree.std():.1f})")
    ax[0].hist(z_phy, bins=np.linspace(-8, 8, 40), density=True, color=COLORS["phylo"],
               alpha=0.55, label=f"phylo contrasts (sd {z_phy.std():.1f})")
    ax[0].plot(grid, stats.norm.pdf(grid), "k--", lw=1.5, label="N(0,1) null")
    ax[0].set_xlabel("gene-gene association z")
    ax[0].set_ylabel("density")
    ax[0].legend(fontsize=8)
    ax[0].set_title("(A) Independent genes on the tree:\nHotspot z is over-dispersed, "
                    "contrasts calibrated", fontsize=10)

    order = ["tip", "knn", "tree", "phylo"]
    vals = [nsig[k] for k in order]
    bars = ax[1].bar(range(4), vals, color=[COLORS[k] for k in order], width=0.65)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([LABELS[k] for k in order], fontsize=8)
    ax[1].set_ylabel(f"# 'significant' gene pairs of {n_pairs}\n(BH-FDR<0.05; truth = 0)")
    ax[1].set_title("(B) One history: spurious gene\nmodules from the tree", fontsize=10)
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=9)

    rvals = [rate[k] for k in order]
    bars = ax[2].bar(range(4), rvals, color=[COLORS[k] for k in order], width=0.65)
    ax[2].axhline(0.05, ls=":", color="grey", label="nominal 0.05")
    ax[2].set_xticks(range(4))
    ax[2].set_xticklabels([LABELS[k] for k in order], fontsize=8)
    ax[2].set_ylabel(f"P(>=1 significant pair), {R} histories")
    ax[2].set_ylim(0, 1.05)
    ax[2].legend(fontsize=8)
    ax[2].set_title("(C) Only the deconfounded estimator\nis calibrated", fontsize=10)
    for b, v in zip(bars, rvals):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGDIR, f"hotspot_confounding_{tumor}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return dict(tumor=tumor, n=n, nsig=nsig, rate=rate, n_pairs=n_pairs)


if __name__ == "__main__":
    import anndata
    ad = anndata.read_h5ad(DATA + "/expression/adata_processed.nt.h5ad")
    run(tumor="3726_NT_T2", adata=ad)
