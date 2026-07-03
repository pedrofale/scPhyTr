"""No-pseudobulk plasticity: scPhyTr separates biological within-clone variation from count noise.

scPhyTr keeps cells as **replicates within a clone** and models raw counts, so it can split a gene's
variance into heritable (between-clone) and *plastic* (within-clone) parts -- a decomposition that
methods collapsing each clone to one value (pseudobulk) structurally cannot compute, and that a
naive within-clone variance on ``log1p(Y/S)`` gets wrong, because at single-cell depth **Poisson/NB
sampling noise masquerades as biological plasticity**. The naive within-clone variance is therefore
inflated -- worse the sparser the data -- while scPhyTr attributes the sampling to the observation
layer.

Ground truth is unambiguous: we plant *rigid* genes (all cells in a clone share the clone's latent
value exactly, so the true within-clone plasticity is ZERO -- every within-clone difference is pure
sampling noise) alongside *plastic* genes (real within-clone spread). A correct method reports ~0
plasticity for the rigid genes at every depth.

  * scPhyTr -- tl.plasticity (count Laplace-EM: heritable K + within-clone NB dispersion).
  * naive   -- within-clone / total variance of log1p(Y/S) (Gaussian, no count model).

Panel A: recovered plasticity of the RIGID genes vs depth (true 0; naive inflates at low depth,
scPhyTr stays flat near 0). Panel B: AUROC separating plastic from rigid genes vs depth.
"""
import os
import numpy as np
import pandas as pd
import anndata as ad

import scphytr as ph
from analysis.benchmark.spatial_decomposition import _tree
from scphytr.tools.heritability import shared_ancestry_cov

OUT = os.path.dirname(__file__)


def _phylo_chol(tree):
    C = np.asarray(shared_ancestry_cov(tree)[0])
    Ccorr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    return np.linalg.cholesky(Ccorr + 1e-8 * np.eye(C.shape[0]))


def _panel(n_rigid=6, n_plastic=6, v_plastic=0.5):
    vplast = np.r_[np.zeros(n_rigid), np.full(n_plastic, v_plastic)]     # rigid: 0 within-clone
    is_plastic = np.r_[np.zeros(n_rigid, int), np.ones(n_plastic, int)]
    names = [f"R{i}" for i in range(n_rigid)] + [f"P{i}" for i in range(n_plastic)]
    return vplast, is_plastic, names


def _auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0 or not np.isfinite(score).all():
        return np.nan
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[label == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def _simulate(tree, Lc, leaves, m, vplast, S, rng, dispersion=None):
    """m cells per leaf. Leaf latent u ~ BM (var 1); cell latent w = u + N(0, vplast_g); counts."""
    nL = len(leaves); p = len(vplast)
    u = Lc @ rng.standard_normal((nL, p))                     # between-clone heritable (var 1)
    Y = []; species = []
    for li, l in enumerate(leaves):
        w = u[li][None, :] + rng.standard_normal((m, p)) * np.sqrt(vplast)[None, :]
        lam = S * np.exp(w)
        Yl = rng.poisson(lam) if dispersion is None else rng.poisson(rng.gamma(dispersion, lam / dispersion))
        Y.append(Yl); species += [l] * m
    return np.asarray(np.vstack(Y), float), np.asarray(species)


def _naive_plasticity(Y, S, species, names):
    """within-clone / total variance of log1p(Y/S), per gene."""
    L = np.log1p(Y / S)
    df = pd.DataFrame(L, columns=names); df["_sp"] = species
    within = df.groupby("_sp").var(ddof=1).mean(axis=0)          # mean within-clone variance
    between = df.groupby("_sp").mean().var(ddof=1, axis=0)       # variance of clone means
    return (within / (within + between)).reindex(names).values


def compute(depths=(1, 2, 4, 8, 16, 32, 64, 128), n=60, m=8, reps=5, dispersion=None, seed0=0):
    tree = _tree(n, 0); leaves = tree.phylotree.get_leaf_names(); Lc = _phylo_chol(tree)
    vplast, is_plastic, names = _panel()
    rows = []
    for S in depths:
        acc = {"scPhyTr_rigid": [], "naive_rigid": [], "scPhyTr_auroc": [], "naive_auroc": [], "mc": []}
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            Y, species = _simulate(tree, Lc, leaves, m, vplast, S, rng, dispersion)
            acc["mc"].append(Y.mean())
            A = ad.AnnData(X=Y.copy()); A.var_names = names
            A.obs["species"] = list(species); A.obs["size_factors"] = np.full(len(species), float(S))
            ph.pp.setup_anndata(A, tree)
            ph.tl.plasticity(A, names, dispersion=(dispersion if dispersion else 10.0))
            psc = A.var["plasticity"].reindex(names).values
            pnv = _naive_plasticity(Y, S, species, names)
            acc["scPhyTr_rigid"].append(np.mean(psc[is_plastic == 0]))
            acc["naive_rigid"].append(np.mean(pnv[is_plastic == 0]))
            acc["scPhyTr_auroc"].append(_auroc(psc, is_plastic))
            acc["naive_auroc"].append(_auroc(pnv, is_plastic))
        row = {"depth": S, "mean_count": np.mean(acc["mc"])}
        for k in ["scPhyTr_rigid", "naive_rigid", "scPhyTr_auroc", "naive_auroc"]:
            row[k] = np.nanmean(acc[k])
        rows.append(row)
        print(f"  depth {S:4d} (mc {row['mean_count']:5.1f}): rigid-gene plasticity (true 0)  "
              f"scPhyTr {row['scPhyTr_rigid']:.2f}  naive {row['naive_rigid']:.2f}  | "
              f"AUROC sc {row['scPhyTr_auroc']:.2f} nv {row['naive_auroc']:.2f}", flush=True)
    return pd.DataFrame(rows)


def figure(df=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None:
        df = pd.read_csv(os.path.join(OUT, "depth_plasticity.csv"))
    colors = {"scPhyTr": "#2c7fb8", "naive": "#e45756"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    # (A) rigid-gene plasticity vs depth (true 0)
    ax[0].plot(df["mean_count"], df["scPhyTr_rigid"], "-o", color=colors["scPhyTr"], label="scPhyTr")
    ax[0].plot(df["mean_count"], df["naive_rigid"], "-o", color=colors["naive"], label="naive (log1p)")
    ax[0].axhline(0.0, ls="--", color="grey", label="true plasticity = 0")
    ax[0].axvspan(1, 6, color="#cccccc", alpha=0.25)
    ax[0].annotate("MERFISH /\nreal PEtracer", (2.4, 0.72), fontsize=8, color="#555")
    ax[0].set_xscale("log"); ax[0].set_xlabel("mean counts per gene per cell (depth)")
    ax[0].set_ylabel("recovered plasticity of RIGID genes"); ax[0].set_ylim(-0.05, 1.0)
    ax[0].set_title("(A) Count noise masquerades as plasticity; scPhyTr isn't fooled")
    ax[0].legend(fontsize=8)
    # (B) AUROC separating plastic from rigid genes
    ax[1].plot(df["mean_count"], df["scPhyTr_auroc"], "-o", color=colors["scPhyTr"], label="scPhyTr")
    ax[1].plot(df["mean_count"], df["naive_auroc"], "-o", color=colors["naive"], label="naive (log1p)")
    ax[1].axhline(0.5, ls=":", color="grey")
    ax[1].set_xscale("log"); ax[1].set_xlabel("mean counts per gene per cell (depth)")
    ax[1].set_ylabel("AUROC (plastic vs rigid genes)"); ax[1].set_ylim(0.4, 1.02)
    ax[1].set_title("(B) Detecting genuinely plastic genes")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "depth_plasticity.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    print("== no-pseudobulk plasticity vs sequencing depth (rigid genes: true plasticity 0) ==")
    df = compute()
    df.to_csv(os.path.join(OUT, "depth_plasticity.csv"), index=False)
    figure(df)
    lo = df.iloc[0]
    print(f"\nAt {lo['mean_count']:.1f} counts/cell: rigid-gene plasticity scPhyTr "
          f"{lo['scPhyTr_rigid']:.2f} vs naive {lo['naive_rigid']:.2f} (true 0) -- naive mistakes "
          f"count noise for biological plasticity.")


if __name__ == "__main__":
    main()
