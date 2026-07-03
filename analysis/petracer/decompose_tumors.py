"""scPhyTr on real PEtracer tumors: heritable-vs-niche variance decomposition + validation.

For each clonal tumour in a PEtracer ``*_tumor_tracing.h5td`` (figshare 28473866) we run
``tl.decompose_variance`` -- the additive tree(BM)+space(GMRF)+iid count model -- to split every
gene's MERFISH expression into a heritable (lineage) and a niche (spatial) component, then check the
split against *independent* spatial evidence:

  * spatial Moran's I of the gene on the leaf kNN graph  (a gene called niche should be spatial),
  * |correlation| with ``tumor_boundary_dist``            (the real tumour-margin gradient).

The decomposition is cached per tumour to ``M{sample}_tree{key}_decompose.csv`` (it is the slow part
-- minutes on the larger trees); the cheap validation statistics are always recomputed. Produces
``figures/petracer_decompose.png``:
  (A) per-tumour frac_heritable distribution,
  (B) validation: frac_heritable vs spatial Moran's I (niche genes cluster at low frac / high I),
  (C) cross-tumour reproducibility of per-gene frac_heritable on shared genes.

Usage:
    PYTHONPATH=src:. python analysis/petracer/decompose_tumors.py [--recompute] [--trees 4,1]
"""
import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

import scphytr as ph
from analysis.petracer.load import load_tumor_tree, DATA

OUT = os.path.dirname(__file__)
FIGDIR = os.path.join(OUT, "figures")


def _csv_path(sample, tree_key):
    return os.path.join(OUT, f"M{sample}_tree{tree_key}_decompose.csv")


def morans_I(z, W):
    """Global Moran's I of leaf-vector ``z`` on the (sparse) spatial weight matrix ``W``."""
    z = z - z.mean()
    s0 = W.sum()
    den = float(z @ z)
    if den <= 0 or s0 <= 0:
        return np.nan
    return float(len(z)) / s0 * float(z @ (W @ z)) / den


def analyze_tumor(sample, tree_key, recompute=False, n_neighbors=8):
    """Return a per-gene DataFrame (frac_heritable, v_phylo, v_space, spatial Moran's I, boundary
    corr) for one tumour. The decomposition is cached; validation stats are always fresh."""
    path = os.path.join(DATA, f"M{sample}_tumor_tracing.h5td")
    A, tree = load_tumor_tree(path, tree_key)
    ph.pp.setup_anndata(A, tree)
    ph.pp.spatial_neighbors(A, n_neighbors=n_neighbors)

    csv = _csv_path(sample, tree_key)
    if os.path.exists(csv) and not recompute:
        dec = pd.read_csv(csv, index_col=0).reindex(A.var_names)
    else:
        print(f"  [M{sample} tree {tree_key}] decomposing {A.n_vars} genes on {A.n_obs} cells "
              f"(cached to {os.path.basename(csv)}) ...", flush=True)
        ph.tl.decompose_variance(A, dispersion=None)
        dec = A.var[["v_phylo", "v_space", "frac_heritable"]].copy()
        dec.to_csv(csv)

    # --- independent validation (cheap: autocorrelation only) ---
    leaves = tree.phylotree.get_leaf_names()
    posl = {s: i for i, s in enumerate(leaves)}
    order = np.array([posl[s] for s in A.obs["species"]])
    X = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
    sf = np.asarray(A.obs["size_factors"], float)
    Lcell = np.log1p(X / np.maximum(sf[:, None], 1e-9))
    Lleaf = np.zeros_like(Lcell); Lleaf[order] = Lcell
    W = A.uns["spatial_graph"]["weights"]
    W = W if sp.issparse(W) else sp.csr_matrix(W)
    mi = np.array([morans_I(Lleaf[:, j], W) for j in range(A.n_vars)])

    bd = pd.to_numeric(A.obs.get("tumor_boundary_dist", pd.Series(index=A.obs_names)), errors="coerce").values
    ok = np.isfinite(bd)
    if ok.sum() > 5:
        bdcorr = np.array([abs(np.corrcoef(Lcell[ok, j], bd[ok])[0, 1])
                           if np.std(Lcell[ok, j]) > 0 else np.nan for j in range(A.n_vars)])
    else:
        bdcorr = np.full(A.n_vars, np.nan)

    df = dec.copy()
    df["moran_I"] = mi
    df["boundary_corr"] = bdcorr
    df["tumor"] = f"M{sample}·{tree_key}"
    df["n_cells"] = A.n_obs
    return df


def _r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 2 else np.nan


def figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    labels = list(results.keys())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (A) per-tumour frac_heritable distribution
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(labels)))
    for c, lab in zip(colors, labels):
        fr = results[lab]["frac_heritable"].values
        fr = fr[np.isfinite(fr)]
        ax[0].hist(fr, bins=np.linspace(0, 1, 21), histtype="step", lw=2, color=c,
                   label=f"{lab} (n={results[lab]['n_cells'].iloc[0]} cells, med {np.median(fr):.2f})")
    ax[0].set_xlabel("frac_heritable  (lineage ← → niche)"); ax[0].set_ylabel("genes")
    ax[0].set_title("(A) Heritable vs niche split, per tumour"); ax[0].legend(fontsize=8)

    # (B) validation: frac_heritable vs spatial Moran's I
    for c, lab in zip(colors, labels):
        d = results[lab]
        ax[1].scatter(d["frac_heritable"], d["moran_I"], s=14, color=c, alpha=0.6, label=lab)
    alld = pd.concat(results.values())
    r = _r(alld["frac_heritable"].values, alld["moran_I"].values)
    ax[1].set_xlabel("frac_heritable"); ax[1].set_ylabel("spatial Moran's I (independent)")
    ax[1].set_title(f"(B) Niche calls track real spatial structure\n(pooled r = {r:.2f}, expected < 0)")
    ax[1].legend(fontsize=8)

    # (C) cross-tumour reproducibility on shared genes (first two tumours)
    if len(labels) >= 2:
        a, b = results[labels[0]], results[labels[1]]
        j = a[["frac_heritable"]].join(b[["frac_heritable"]], lsuffix="_a", rsuffix="_b").dropna()
        rr = _r(j["frac_heritable_a"].values, j["frac_heritable_b"].values)
        ax[2].scatter(j["frac_heritable_a"], j["frac_heritable_b"], s=14, color="#444")
        # a few story genes: reproducibly niche (Arg1), a flipper (Nes), reproducibly heritable (Krt79)
        for g, dx, dy, col in [("Arg1", 0.02, 0.03, "#c0392b"), ("Sdc1", 0.02, -0.06, "#c0392b"),
                               ("Nes", -0.10, 0.03, "#e67e22"), ("Krt79", -0.14, -0.02, "#2c7fb8")]:
            if g in j.index:
                ax[2].annotate(g, (j.loc[g, "frac_heritable_a"], j.loc[g, "frac_heritable_b"]),
                               xytext=(j.loc[g, "frac_heritable_a"] + dx, j.loc[g, "frac_heritable_b"] + dy),
                               fontsize=8, color=col, fontweight="bold")
        ax[2].plot([0, 1], [0, 1], ls=":", color="grey")
        ax[2].set_xlabel(f"frac_heritable · {labels[0]}"); ax[2].set_ylabel(f"frac_heritable · {labels[1]}")
        ax[2].set_title(f"(C) Cross-tumour reproducibility\n(r = {rr:.2f}, {len(j)} shared genes)")
    else:
        ax[2].axis("off")

    fig.tight_layout()
    out = os.path.join(FIGDIR, "petracer_decompose.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def report(results):
    for lab, d in results.items():
        fr = d["frac_heritable"].values
        o = np.argsort(-np.nan_to_num(fr, nan=-1))
        r_mi = _r(d["frac_heritable"].values, d["moran_I"].values)
        r_bd = _r(d["frac_heritable"].values, d["boundary_corr"].values)
        print(f"\n== {lab}  ({d['n_cells'].iloc[0]} cells) ==")
        print(f"  frac_heritable median {np.nanmedian(fr):.2f}  IQR[{np.nanpercentile(fr,25):.2f},{np.nanpercentile(fr,75):.2f}]")
        print(f"  validation: vs Moran's I r={r_mi:.2f} | vs |boundary corr| r={r_bd:.2f}  (both expected < 0)")
        print("  top niche: " + ", ".join(d.index[o[-8:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="2", help="tumour sample id (default 2 = M2)")
    ap.add_argument("--trees", default="4,1", help="comma-separated tree keys, e.g. 4,1")
    ap.add_argument("--recompute", action="store_true", help="recompute the (cached) decomposition")
    args = ap.parse_args()
    results = {}
    for tk in args.trees.split(","):
        d = analyze_tumor(args.sample, tk.strip(), recompute=args.recompute)
        results[d["tumor"].iloc[0]] = d
    report(results)
    figure(results)


if __name__ == "__main__":
    main()
