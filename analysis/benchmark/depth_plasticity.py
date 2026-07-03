"""No-pseudobulk plasticity: scPhyTr recovers within-clone variation; a naive estimate inflates it.

scPhyTr keeps cells as replicates within a clone and models raw counts, so it splits a gene's
variance into heritable (between-clone) and *plastic* (within-clone). A naive within-clone variance
on ``log1p(Y/S)`` adds the Poisson/NB **sampling** variance to the biological within-clone spread, so
it OVER-estimates plasticity -- worse the sparser the data. (Pseudobulk methods cannot estimate it at
all.) scPhyTr attributes the sampling to the observation layer.

We measure *recovery of a continuous ground truth* -- NOT a binary detection (an earlier binary AUROC
was near-tautological: it pitted exactly-Poisson genes, i.e. scPhyTr's null, against overdispersed
ones, which the overdispersion estimator separates perfectly at any depth). Here true plasticity
``tp = V_plast/(V_plast+V_herit)`` is swept continuously; counts are normalised so the mean count per
gene is ~S regardless of variance (so the depth axis is honest).

  * scPhyTr -- tl.plasticity (count Laplace-EM: heritable K + within-clone NB dispersion).
  * naive   -- within-clone / total variance of log1p(Y/S).

Panel A: estimated vs true plasticity at MERFISH depth (scPhyTr near the diagonal but imperfect;
naive shifted up). Panel B: bias (estimated - true) vs depth at a fixed true plasticity (naive's
inflation decays with depth; scPhyTr ~flat).
"""
import os
import numpy as np
import pandas as pd
import anndata as ad

import scphytr as ph
from analysis.benchmark.spatial_decomposition import _tree
from scphytr.tools.heritability import shared_ancestry_cov

OUT = os.path.dirname(__file__)
V_HERIT = 1.0                                   # between-clone (heritable) latent log-variance


def _phylo_chol(tree):
    C = np.asarray(shared_ancestry_cov(tree)[0])
    Ccorr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    return np.linalg.cholesky(Ccorr + 1e-8 * np.eye(C.shape[0]))


def _simulate(tree, Lc, leaves, m, vplast, S, rng):
    """m cells/leaf. Leaf latent u~BM (var V_HERIT); cell latent w=u+N(0,vplast_g). Counts are
    normalised per gene so the mean count is ~S at every plasticity level (honest depth axis)."""
    nL = len(leaves); p = len(vplast)
    u = Lc @ rng.standard_normal((nL, p)) * np.sqrt(V_HERIT)
    W = np.empty((nL * m, p)); species = []
    for li, l in enumerate(leaves):
        W[li * m:(li + 1) * m] = u[li][None, :] + rng.standard_normal((m, p)) * np.sqrt(vplast)[None, :]
        species += [l] * m
    E = np.exp(W)
    lam = S * E / E.mean(axis=0, keepdims=True)          # mean count per gene == S
    return rng.poisson(lam).astype(float), np.asarray(species)


def _naive_plasticity(Y, S, species, names):
    L = np.log1p(Y / S)
    df = pd.DataFrame(L, columns=names); df["_sp"] = species
    within = df.groupby("_sp").var(ddof=1).mean(axis=0)
    between = df.groupby("_sp").mean().var(ddof=1, axis=0)
    return (within / (within + between)).reindex(names).values


def _estimate(tree, leaves, m, vplast, S, rng):
    names = [f"g{i}" for i in range(len(vplast))]
    Y, species = _simulate(tree, Lc_GLOBAL, leaves, m, vplast, S, rng)
    A = ad.AnnData(X=Y.copy()); A.var_names = names
    A.obs["species"] = list(species); A.obs["size_factors"] = np.full(len(species), float(S))
    ph.pp.setup_anndata(A, tree)
    ph.tl.plasticity(A, names, dispersion=10.0)
    return A.var["plasticity"].reindex(names).values, _naive_plasticity(Y, S, species, names)


Lc_GLOBAL = None


def _tp_to_vplast(tp):
    return tp / np.maximum(1.0 - tp, 1e-6) * V_HERIT


def recovery(depth=3, tps=(0.0, 0.1, 0.2, 0.35, 0.5, 0.65), n=60, m=8, ng=5, reps=4, seed0=0):
    global Lc_GLOBAL
    tree = _tree(n, 0); leaves = tree.phylotree.get_leaf_names(); Lc_GLOBAL = _phylo_chol(tree)
    rows = []
    for tp in tps:
        vplast = np.full(ng, _tp_to_vplast(tp))
        sc, nv = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            psc, pnv = _estimate(tree, leaves, m, vplast, depth, rng)
            sc.append(np.mean(psc)); nv.append(np.mean(pnv))
        rows.append({"true_plasticity": tp, "scPhyTr": np.mean(sc), "naive": np.mean(nv)})
        print(f"  true {tp:.2f}: scPhyTr {np.mean(sc):.2f}  naive {np.mean(nv):.2f}", flush=True)
    return pd.DataFrame(rows)


def depth_bias(true_plasticity=0.35, depths=(1, 2, 4, 8, 16, 32, 64), n=60, m=8, ng=5, reps=4, seed0=0):
    global Lc_GLOBAL
    tree = _tree(n, 0); leaves = tree.phylotree.get_leaf_names(); Lc_GLOBAL = _phylo_chol(tree)
    vp = _tp_to_vplast(true_plasticity)
    rows = []
    for S in depths:
        sc, nv, mc = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            vplast = np.full(ng, vp)
            Y, species = _simulate(tree, Lc_GLOBAL, leaves, m, vplast, S, rng)
            mc.append(Y.mean())
            psc, pnv = _estimate(tree, leaves, m, vplast, S, rng)
            sc.append(np.mean(psc)); nv.append(np.mean(pnv))
        rows.append({"depth": S, "mean_count": np.mean(mc), "true_plasticity": true_plasticity,
                     "scPhyTr": np.mean(sc), "naive": np.mean(nv)})
        print(f"  depth {S:3d} (mc {np.mean(mc):5.1f}): scPhyTr {np.mean(sc):.2f}  "
              f"naive {np.mean(nv):.2f}  (true {true_plasticity})", flush=True)
    return pd.DataFrame(rows)


def figure(rec=None, dep=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rec is None:
        rec = pd.read_csv(os.path.join(OUT, "depth_plasticity_recovery.csv"))
    if dep is None:
        dep = pd.read_csv(os.path.join(OUT, "depth_plasticity_depthbias.csv"))
    colors = {"scPhyTr": "#2c7fb8", "naive": "#e45756"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    # (A) estimated vs true at MERFISH depth
    for m in ["scPhyTr", "naive"]:
        ax[0].plot(rec["true_plasticity"], rec[m], "-o", color=colors[m], label=m)
    ax[0].plot([0, 0.7], [0, 0.7], ls=":", color="grey", label="truth")
    ax[0].set_xlabel("true plasticity"); ax[0].set_ylabel("estimated plasticity (~3 counts/cell)")
    ax[0].set_title("(A) Recovery at MERFISH depth: naive inflated by count noise")
    ax[0].legend(fontsize=8)
    # (B) bias vs depth at fixed true plasticity
    tp = dep["true_plasticity"].iloc[0]
    for m in ["scPhyTr", "naive"]:
        ax[1].plot(dep["mean_count"], dep[m] - tp, "-o", color=colors[m], label=m)
    ax[1].axhline(0.0, ls="--", color="grey")
    ax[1].axvspan(1, 6, color="#cccccc", alpha=0.25)
    ax[1].annotate("MERFISH", (2.2, 0.28), fontsize=8, color="#555")
    ax[1].set_xscale("log"); ax[1].set_xlabel("mean counts per gene per cell (depth)")
    ax[1].set_ylabel(f"bias  (estimated - true, true={tp})")
    ax[1].set_title("(B) Naive plasticity is inflated at low depth; scPhyTr ~unbiased")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "depth_plasticity.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    print("== (A) plasticity recovery vs true, at MERFISH depth (~3 counts) ==")
    rec = recovery()
    rec.to_csv(os.path.join(OUT, "depth_plasticity_recovery.csv"), index=False)
    print("== (B) plasticity bias vs sequencing depth (true plasticity 0.35) ==")
    dep = depth_bias()
    dep.to_csv(os.path.join(OUT, "depth_plasticity_depthbias.csv"), index=False)
    figure(rec, dep)


if __name__ == "__main__":
    main()
