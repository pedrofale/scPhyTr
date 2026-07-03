"""Count-native gene-gene co-evolution resists the correlation attenuation of noiseless-trait PCM.

Measurement error attenuates correlations: an observed correlation is the true correlation times the
reliability of each variable, and at single-cell / MERFISH depth the reliability of ``log1p(Y/S)`` is
low. So the field-standard phylogenetic correlation estimators -- Felsenstein's independent contrasts
and phylogenetic GLS, which assume the leaf traits are observed **without error** -- report gene-gene
evolutionary correlations biased toward zero, the more so the sparser the data. scPhyTr fits the
correlated Brownian motion through the count likelihood, so the count noise is in the observation
layer and the latent (evolutionary) correlation is recovered.

We plant a panel of genes co-evolving by **correlated Brownian motion** on the tree: two modules
whose within-module evolutionary correlation is ``rho`` (default 0.7) and whose cross-module
correlation is 0. We observe Poisson/NB counts across a depth sweep and estimate the gene-gene
correlation three ways:

  * scPhyTr   -- tl.evolutionary_correlation (multivariate count Laplace-EM -> uns['K_corr']).
  * contrasts -- Felsenstein's independent-contrasts correlation on log1p(Y/S) (noiseless-trait PCM).
  * naive     -- Pearson correlation of log1p(Y/S) (no tree, no error model).

Panel A: recovered within-module correlation vs depth (scPhyTr near rho; contrasts/naive attenuate).
Panel B: AUROC separating true co-evolving pairs from null pairs vs depth.
"""
import os
import numpy as np
import pandas as pd
import anndata as ad

import scphytr as ph
from analysis.benchmark.spatial_decomposition import _tree
from scphytr.tools.heritability import shared_ancestry_cov
from analysis.kptracer.hotspot_utils import contrast_corr, tip_corr

OUT = os.path.dirname(__file__)


def _phylo_chol(tree):
    Ctup = shared_ancestry_cov(tree)
    C = np.asarray(Ctup[0])
    Ccorr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    return np.linalg.cholesky(Ccorr + 1e-8 * np.eye(C.shape[0])), C


def _panel(nA=3, nB=3, rho=0.7):
    """Two co-evolving modules: within-module evolutionary correlation rho, cross-module 0."""
    p = nA + nB
    R = np.eye(p)
    R[:nA, :nA] = rho; R[nA:, nA:] = rho
    np.fill_diagonal(R, 1.0)
    groups = np.array(["A"] * nA + ["B"] * nB)
    names = [f"A{i}" for i in range(nA)] + [f"B{i}" for i in range(nB)]
    return R, groups, names


def _pairs(groups):
    p = len(groups); iu = np.triu_indices(p, 1)
    same = (groups[iu[0]] == groups[iu[1]]).astype(int)     # true co-evolving pair
    return iu, same


def _auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[label == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def _simulate_counts(Lc, nL, R, S, rng, dispersion=None):
    """Correlated-BM leaf latent Z (nL x p), then Poisson/NB counts at depth S."""
    p = R.shape[0]
    cR = np.linalg.cholesky(R + 1e-9 * np.eye(p))
    Z = (Lc @ rng.standard_normal((nL, p))) @ cR.T          # Cov(Z[:,g],Z[:,h]) = R[g,h]·C_tree
    lam = S * np.exp(Z)
    if dispersion is None:
        return rng.poisson(lam).astype(float)
    return rng.poisson(rng.gamma(dispersion, lam / dispersion)).astype(float)


def compute(depths=(1, 2, 4, 8, 16, 32, 64, 128), rho=0.7, n=70, reps=5, dispersion=None, seed0=0):
    tree = _tree(n, 0); leaves = tree.phylotree.get_leaf_names(); nL = len(leaves)
    Lc, C = _phylo_chol(tree)
    R, groups, names = _panel(rho=rho)
    iu, same = _pairs(groups)
    rows = []
    for S in depths:
        acc = {m: {"within": [], "cross": [], "auroc": []} for m in ["scPhyTr", "contrasts", "naive"]}
        meanct = []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            Y = _simulate_counts(Lc, nL, R, S, rng, dispersion)     # nL x p, leaf order
            meanct.append(Y.mean())
            L = np.log1p(Y / S)
            # scPhyTr count-native
            A = ad.AnnData(X=Y.copy())
            A.var_names = names; A.obs_names = leaves
            A.obs["species"] = list(leaves); A.obs["size_factors"] = np.full(nL, float(S))
            ph.pp.setup_anndata(A, tree)
            ph.tl.evolutionary_correlation(A, names, dispersion=dispersion)
            Ksc = np.abs(A.uns["K_corr"])
            Kct = np.abs(np.nan_to_num(contrast_corr(L, C)[0]))
            Knv = np.abs(np.nan_to_num(tip_corr(L)[0]))
            for m, K in [("scPhyTr", Ksc), ("contrasts", Kct), ("naive", Knv)]:
                v = K[iu]
                acc[m]["within"].append(np.mean(v[same == 1]))
                acc[m]["cross"].append(np.mean(v[same == 0]))
                acc[m]["auroc"].append(_auroc(v, same))
        row = {"depth": S, "mean_count": np.mean(meanct), "true_rho": rho}
        for m in ["scPhyTr", "contrasts", "naive"]:
            row[f"{m}_within"] = np.mean(acc[m]["within"])
            row[f"{m}_cross"] = np.mean(acc[m]["cross"])
            row[f"{m}_auroc"] = np.nanmean(acc[m]["auroc"])
        rows.append(row)
        print(f"  depth {S:4d} (mc {row['mean_count']:5.1f}): within-module corr  "
              f"scPhyTr {row['scPhyTr_within']:.2f}  contrasts {row['contrasts_within']:.2f}  "
              f"naive {row['naive_within']:.2f}   (true {rho})", flush=True)
    return pd.DataFrame(rows)


def figure(df=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None:
        df = pd.read_csv(os.path.join(OUT, "depth_coevolution.csv"))
    colors = {"scPhyTr": "#2c7fb8", "contrasts": "#e6a817", "naive": "#e45756"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    rho = df["true_rho"].iloc[0]
    # (A) recovered within-module evolutionary correlation vs depth
    for m in ["scPhyTr", "contrasts", "naive"]:
        ax[0].plot(df["mean_count"], df[f"{m}_within"], "-o", color=colors[m], label=m)
    ax[0].axhline(rho, ls="--", color="grey", label=f"true ρ = {rho:g}")
    ax[0].axvspan(1, 6, color="#cccccc", alpha=0.25)
    ax[0].annotate("MERFISH /\nreal PEtracer", (2.4, 0.05), fontsize=8, color="#555")
    ax[0].set_xscale("log"); ax[0].set_xlabel("mean counts per gene per cell (depth)")
    ax[0].set_ylabel("recovered within-module correlation"); ax[0].set_ylim(0, 1)
    ax[0].set_title("(A) Noiseless-trait PCM attenuates co-evolution; scPhyTr holds")
    ax[0].legend(fontsize=8)
    # (B) AUROC separating co-evolving from null pairs vs depth
    for m in ["scPhyTr", "contrasts", "naive"]:
        ax[1].plot(df["mean_count"], df[f"{m}_auroc"], "-o", color=colors[m], label=m)
    ax[1].axhline(0.5, ls=":", color="grey")
    ax[1].set_xscale("log"); ax[1].set_xlabel("mean counts per gene per cell (depth)")
    ax[1].set_ylabel("AUROC (co-evolving vs null pairs)"); ax[1].set_ylim(0.4, 1.02)
    ax[1].set_title("(B) Detecting co-evolving gene pairs")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "depth_coevolution.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    print("== gene-gene co-evolution vs sequencing depth (true ρ = 0.7) ==")
    df = compute()
    df.to_csv(os.path.join(OUT, "depth_coevolution.csv"), index=False)
    figure(df)
    lo = df.iloc[0]
    print(f"\nAt {lo['mean_count']:.1f} counts/cell: within-module correlation scPhyTr "
          f"{lo['scPhyTr_within']:.2f} vs contrasts {lo['contrasts_within']:.2f} vs naive "
          f"{lo['naive_within']:.2f} (true 0.70) -- noiseless-trait PCM attenuates co-evolution.")


if __name__ == "__main__":
    main()
