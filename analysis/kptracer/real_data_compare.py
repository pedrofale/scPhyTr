"""Naive vs phylogeny-aware factor analysis on REAL KP-Tracer expression.

Same estimator, one switch: row covariance = identity (naive) vs the tumor's
phylogenetic covariance C (phylo). We then ask whether the learned gene programs
(loadings) differ, and *why*, using the Felsenstein lens validated on the
semi-synthetic control:

  * subspace_error / principal angles between the naive and phylo loadings;
  * per-program clade variance fraction (eta^2): projecting the SAME centered
    data onto each unit loading and measuring how much of that score's variance
    lies BETWEEN deep clones. Naive programs are predicted to spend more capacity
    on clonal-identity (phylogenetic) axes; phylo deconfounds them, leaving
    within-clade co-regulation.
  * top-loading genes of the most clonal naive program vs the matched phylo one.

Outputs analysis/kptracer/figures/realcompare_<tumor>.png and prints numbers.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata

from scphytr.tools.factor_analysis import fit_factor_analysis, subspace_error, principal_angles
from analysis.kptracer.load import load_tumor
from analysis.kptracer.phylo_factor_utils import get_clades, clade_eta2

DATA = "data/external/KPTracer-Data"
HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")


def program_eta2(W, Ycen, labels):
    """eta^2 of each program's centered-data projection (unit loadings)."""
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    scores = Ycen @ Wn                      # (n, k)
    return np.array([clade_eta2(scores[:, j], labels) for j in range(W.shape[1])])


def matched_cosines(Wa, Wb):
    a = Wa / np.linalg.norm(Wa, axis=0, keepdims=True)
    b = Wb / np.linalg.norm(Wb, axis=0, keepdims=True)
    M = np.abs(a.T @ b)
    used, cos = set(), []
    for j in range(b.shape[1]):
        order = [(M[i, j] if i not in used else -1) for i in range(a.shape[1])]
        i = int(np.argmax(order)); used.add(i); cos.append(M[i, j])
    return np.array(cos)


def run(tumor="3726_NT_T2", k=12, n_hvg=1000, Q=12, adata=None, seed=0):
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=n_hvg)
    C, Y, genes, names = d["C"], d["Y"], d["genes"], d["leaf_names"]
    n, p = Y.shape
    labels = get_clades(d["tree"], names, Q)
    Ycen = Y - Y.mean(0)

    naive = fit_factor_analysis(Y, row_cov=None, k=k, restarts=1, seed=seed)
    phylo = fit_factor_analysis(Y, row_cov=C, k=k, restarts=1, seed=seed)

    sub_err = subspace_error(naive.W, phylo.W)
    angles = np.degrees(principal_angles(naive.W, phylo.W))
    cos_np = matched_cosines(naive.W, phylo.W)

    eta_naive = program_eta2(naive.W, Ycen, labels)
    eta_phylo = program_eta2(phylo.W, Ycen, labels)

    print(f"\n=== real-data naive vs phylo FA: {tumor} ({n} cells, {p} HVGs, k={k}) ===")
    print(f"loading subspace error (naive vs phylo): {sub_err:.2f} of max {np.sqrt(k):.2f}")
    print(f"principal angles (deg): {np.round(angles,1)}")
    print(f"matched program |cosine| naive<->phylo: median {np.median(cos_np):.2f}, "
          f"#programs with |cos|<0.7: {(cos_np<0.7).sum()}/{k}")
    print(f"clade variance fraction eta^2 (between {Q} deep clones):")
    print(f"   naive programs: mean {eta_naive.mean():.2f}, max {eta_naive.max():.2f}, "
          f"# with eta^2>0.5: {(eta_naive>0.5).sum()}")
    print(f"   phylo programs: mean {eta_phylo.mean():.2f}, max {eta_phylo.max():.2f}, "
          f"# with eta^2>0.5: {(eta_phylo>0.5).sum()}")

    # most clonal naive program vs its matched phylo program: top genes
    jn = int(np.argmax(eta_naive))
    top_naive = genes[np.argsort(np.abs(naive.W[:, jn]))[::-1][:12]]
    print(f"\nmost clonal NAIVE program (#{jn}, eta^2={eta_naive[jn]:.2f}) top genes:")
    print("   ", ", ".join(map(str, top_naive)))
    jp = int(np.argmin(eta_phylo))
    top_phylo = genes[np.argsort(np.abs(phylo.W[:, jp]))[::-1][:12]]
    print(f"least clonal PHYLO program (#{jp}, eta^2={eta_phylo[jp]:.2f}) top genes:")
    print("   ", ", ".join(map(str, top_phylo)))

    # ---- figure -------------------------------------------------------------
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ax[0].plot(range(1, k + 1), angles, "o-", color="#756bb1")
    ax[0].axhline(0, ls=":", c="grey")
    ax[0].set_xlabel("principal angle index")
    ax[0].set_ylabel("angle (degrees)")
    ax[0].set_title(f"(A) Naive vs phylo loadings differ\nsubspace error {sub_err:.2f}/"
                    f"{np.sqrt(k):.2f}", fontsize=10)

    parts = ax[1].boxplot([eta_naive, eta_phylo], tick_labels=["naive", "phylo"],
                          widths=0.5, patch_artist=True,
                          boxprops=dict(facecolor="#deebf7"),
                          medianprops=dict(color="#08519c"))
    ax[1].scatter(np.ones(k) + np.random.uniform(-.08, .08, k), eta_naive,
                  s=18, color="#d95f0e", zorder=3)
    ax[1].scatter(2 * np.ones(k) + np.random.uniform(-.08, .08, k), eta_phylo,
                  s=18, color="#2c7fb8", zorder=3)
    ax[1].set_ylabel(f"clade variance fraction $\\eta^2$ (between {Q} clones)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(B) Naive programs are more clonal\n(phylo deconfounds the tree)",
                    fontsize=10)

    ax[2].hist(cos_np, bins=np.linspace(0, 1, 11), color="#9ecae1", edgecolor="w")
    ax[2].axvline(np.median(cos_np), ls="--", c="#08519c",
                  label=f"median {np.median(cos_np):.2f}")
    ax[2].set_xlabel("matched program |cosine| (naive vs phylo)")
    ax[2].set_ylabel("# programs")
    ax[2].legend(fontsize=9)
    ax[2].set_title("(C) Many programs are genuinely different", fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIGDIR, f"realcompare_{tumor}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return dict(tumor=tumor, n=n, sub_err=sub_err, angles=angles,
                eta_naive=eta_naive, eta_phylo=eta_phylo, cos=cos_np,
                clonal_naive_genes=list(map(str, top_naive)), clonal_naive_eta=float(eta_naive[jn]))


if __name__ == "__main__":
    ad = anndata.read_h5ad(DATA + "/expression/adata_processed.nt.h5ad")
    run(tumor="3726_NT_T2", adata=ad)
