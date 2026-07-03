"""Count-native heritability resists the count-noise attenuation that cripples Gaussian PCM.

Existing single-cell heritability tools treat expression as a **Gaussian trait on log-normalized
counts** (Pagel's lambda / phylogenetic autocorrelation as in PATH, and what PEtracer uses). At
single-cell / MERFISH sequencing depth (a few counts per gene per cell) that is a measurement-error
problem: Poisson/NB sampling noise in ``log1p(Y/S)`` looks like *non-heritable* trait variance, so
the estimated heritability is **attenuated** -- biased toward zero, and worse the sparser the data.
scPhyTr models the raw counts, so the sampling noise is absorbed by the observation layer and the
latent heritability is recovered.

We plant a leaf trait ``z = u + e`` with a known heritability ``lambda = Var(u)/(Var(u)+Var(e))``
(``u`` = Brownian motion on the tree, ``e`` = i.i.d. leaf residual), observe Poisson/NB counts at a
range of sequencing depths, and estimate ``lambda`` three ways:

  * scPhyTr    -- count-native additive BM + i.i.d. partition (decompose with an identity spatial
                 precision, i.e. the spatial slot is an i.i.d. residual), on raw counts.
  * Gaussian   -- the SAME estimator but fed ``log1p(Y/S)`` as a Gaussian trait (isolates the
                 observation model: same machinery, only the likelihood differs).
  * Pagel      -- Pagel's lambda REML on ``log1p(Y/S)`` (the literal field-standard competitor).

Panel A: estimated lambda vs depth (scPhyTr flat near truth; Gaussian/Pagel attenuate at low depth).
Panel B: calibration at a fixed MERFISH-like depth -- estimated vs true lambda across a grid
(scPhyTr near the diagonal; Gaussian/Pagel compressed toward zero).
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.benchmark.spatial_decomposition import _tree
from scphytr.inference.spatial_decomposition import decompose, GaussianLeafObservation
from scphytr.observation_models import SubclonalObservation
from scphytr.tools.heritability import shared_ancestry_cov, pagels_lambda

OUT = os.path.dirname(__file__)


def _phylo_chol(tree):
    """Cholesky of the unit-marginal phylogenetic correlation (to draw BM leaf values)."""
    Ctup = shared_ancestry_cov(tree)
    C = np.asarray(Ctup[0])
    Ccorr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    return np.linalg.cholesky(Ccorr + 1e-8 * np.eye(C.shape[0])), Ctup


def _simulate(Lc, nL, true_lambda, S, rng, dispersion=None):
    """Counts for one gene at depth S: z = u(BM) + e(iid), Y ~ Poisson/NB(S·exp(z))."""
    u = Lc @ rng.standard_normal(nL) * np.sqrt(true_lambda)
    e = rng.standard_normal(nL) * np.sqrt(1.0 - true_lambda)
    lam = S * np.exp(u + e)
    if dispersion is None:
        return rng.poisson(lam).astype(float)
    return rng.poisson(rng.gamma(dispersion, lam / dispersion)).astype(float)


def _estimates(tree, Y, S, Qs, Ctup, leaves, dispersion=None):
    nL = len(leaves)
    idx = np.arange(nL); sf = np.full(nL, float(S))
    sc = decompose(tree, SubclonalObservation(Y, sf, idx, nL, dispersion=dispersion),
                   Qs, include_residual=False).frac_heritable
    Lx = np.log1p(Y / S)
    ga = decompose(tree, GaussianLeafObservation(Lx - Lx.mean(), noise=1e-2),
                   Qs, include_residual=False).frac_heritable
    pg = pagels_lambda(tree, {leaves[i]: float(Lx[i]) for i in range(nL)}, C=Ctup)["lambda"]
    return sc, ga, pg


def depth_sweep(depths=(1, 2, 4, 8, 16, 32, 64, 128), true_lambda=0.6, n=90, reps=6, ng=12,
                dispersion=None, seed0=0):
    tree = _tree(n, 0); leaves = tree.phylotree.get_leaf_names(); nL = len(leaves)
    Lc, Ctup = _phylo_chol(tree); Qs = sp.identity(nL, format="csc")
    rows = []
    for S in depths:
        acc = {"scPhyTr": [], "Gaussian": [], "Pagel": [], "meanct": []}
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            for _ in range(ng):
                Y = _simulate(Lc, nL, true_lambda, S, rng, dispersion)
                acc["meanct"].append(Y.mean())
                sc, ga, pg = _estimates(tree, Y, S, Qs, Ctup, leaves, dispersion)
                acc["scPhyTr"].append(sc); acc["Gaussian"].append(ga); acc["Pagel"].append(pg)
        row = {"depth": S, "mean_count": np.mean(acc["meanct"]), "true_lambda": true_lambda}
        for m in ["scPhyTr", "Gaussian", "Pagel"]:
            row[f"{m}_mean"] = np.mean(acc[m]); row[f"{m}_sd"] = np.std(acc[m])
        rows.append(row)
        print(f"  depth {S:4d} (mean count {row['mean_count']:5.1f}): "
              f"scPhyTr {row['scPhyTr_mean']:.2f}  Gaussian {row['Gaussian_mean']:.2f}  "
              f"Pagel {row['Pagel_mean']:.2f}", flush=True)
    return pd.DataFrame(rows)


def calibration(depth=3, true_lambdas=(0.1, 0.3, 0.5, 0.7, 0.9), n=90, reps=6, ng=10,
                dispersion=None, seed0=0):
    tree = _tree(n, 0); leaves = tree.phylotree.get_leaf_names(); nL = len(leaves)
    Lc, Ctup = _phylo_chol(tree); Qs = sp.identity(nL, format="csc")
    rows = []
    for tl in true_lambdas:
        acc = {"scPhyTr": [], "Gaussian": [], "Pagel": []}
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            for _ in range(ng):
                Y = _simulate(Lc, nL, tl, depth, rng, dispersion)
                sc, ga, pg = _estimates(tree, Y, depth, Qs, Ctup, leaves, dispersion)
                acc["scPhyTr"].append(sc); acc["Gaussian"].append(ga); acc["Pagel"].append(pg)
        row = {"true_lambda": tl, "depth": depth}
        for m in ["scPhyTr", "Gaussian", "Pagel"]:
            row[f"{m}_mean"] = np.mean(acc[m]); row[f"{m}_sd"] = np.std(acc[m])
        rows.append(row)
        print(f"  true λ {tl:.1f}: scPhyTr {row['scPhyTr_mean']:.2f}  "
              f"Gaussian {row['Gaussian_mean']:.2f}  Pagel {row['Pagel_mean']:.2f}", flush=True)
    return pd.DataFrame(rows)


def figure(sweep=None, calib=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if sweep is None:
        sweep = pd.read_csv(os.path.join(OUT, "depth_heritability_sweep.csv"))
    if calib is None:
        calib = pd.read_csv(os.path.join(OUT, "depth_heritability_calibration.csv"))
    colors = {"scPhyTr": "#2c7fb8", "Gaussian": "#e6a817", "Pagel": "#e45756"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    # (A) heritability vs depth
    tl = sweep["true_lambda"].iloc[0]
    for m in ["scPhyTr", "Gaussian", "Pagel"]:
        ax[0].plot(sweep["mean_count"], sweep[f"{m}_mean"], "-o", color=colors[m], label=m)
        ax[0].fill_between(sweep["mean_count"], sweep[f"{m}_mean"] - sweep[f"{m}_sd"],
                           sweep[f"{m}_mean"] + sweep[f"{m}_sd"], color=colors[m], alpha=0.12)
    ax[0].axhline(tl, ls="--", color="grey", label=f"true λ = {tl:g}")
    ax[0].axvspan(1, 6, color="#cccccc", alpha=0.25)
    ax[0].annotate("MERFISH /\nreal PEtracer", (2.4, 0.06), fontsize=8, color="#555")
    ax[0].set_xscale("log"); ax[0].set_xlabel("mean counts per gene per cell (sequencing depth)")
    ax[0].set_ylabel("estimated heritability λ"); ax[0].set_ylim(0, 1)
    ax[0].set_title("(A) Gaussian PCM attenuates at low depth; scPhyTr holds")
    ax[0].legend(fontsize=8)

    # (B) calibration at fixed low depth
    d = calib["depth"].iloc[0]
    for m in ["scPhyTr", "Gaussian", "Pagel"]:
        ax[1].errorbar(calib["true_lambda"], calib[f"{m}_mean"], yerr=calib[f"{m}_sd"],
                       fmt="-o", color=colors[m], label=m, capsize=3)
    ax[1].plot([0, 1], [0, 1], ls=":", color="grey")
    ax[1].set_xlabel("true λ"); ax[1].set_ylabel("estimated λ"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    ax[1].set_title(f"(B) Calibration at MERFISH depth (~{calib['scPhyTr_mean'].size and d} counts)")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "depth_heritability.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    print("== (A) heritability vs sequencing depth (true λ = 0.6) ==")
    sweep = depth_sweep()
    sweep.to_csv(os.path.join(OUT, "depth_heritability_sweep.csv"), index=False)
    print("== (B) calibration at MERFISH depth (~3 counts) ==")
    calib = calibration()
    calib.to_csv(os.path.join(OUT, "depth_heritability_calibration.csv"), index=False)
    figure(sweep, calib)
    lo = sweep.iloc[0]
    print(f"\nAt {lo['mean_count']:.1f} counts/cell: scPhyTr {lo['scPhyTr_mean']:.2f} vs "
          f"Gaussian {lo['Gaussian_mean']:.2f} vs Pagel {lo['Pagel_mean']:.2f} (true 0.60) -- "
          f"the Gaussian methods lose ~half the heritable signal to count noise.")


if __name__ == "__main__":
    main()
