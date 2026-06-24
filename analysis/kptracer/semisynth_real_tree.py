"""Semi-synthetic sanity check on a REAL KP-Tracer tumor phylogeny.

Ground-truth control for the real-data analysis: we keep the real tumor tree
topology/branch lengths but *simulate* expression so we know the answer. Genes
are evolutionarily independent (no true gene program). We then run the same
"is there a factor?" test (Horn's parallel analysis) on the raw leaves (naive)
and on Felsenstein's independent contrasts (phylo), and over many simulated
histories measure how often each invents a factor.

If naive factor analysis routinely finds a spurious top program -- one aligned
with the tumor's deep subclonal split -- while the phylo test does not, then the
Felsenstein confounding is operative on real tumor tree shapes, and any program
difference we later see on real data has a principled cause.

Outputs analysis/kptracer/figures/semisynth_<tumor>.png and prints the numbers.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata

from analysis.kptracer.load import load_tumor
from analysis.kptracer.phylo_factor_utils import (
    chol, simulate_independent_genes, phylo_contrasts, parallel_analysis,
    deep_clade_indicator)

DATA = "data/external/KPTracer-Data"
HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")


def run(tumor="3726_NT_T2", p=20, R=200, adata=None, seed=0):
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=1)   # expression unused here
    C = d["C"]
    names = d["leaf_names"]
    n = len(names)
    L_C = chol(C)
    clade = deep_clade_indicator(d["tree"], names)

    # ---- one representative dataset -----------------------------------------
    rng = np.random.default_rng(seed)
    Y = simulate_independent_genes(L_C, p, rng)
    obs_n, null_n, sig_n = parallel_analysis(Y, 300, np.random.default_rng(1))
    Yc = phylo_contrasts(Y, C)
    obs_p, null_p, sig_p = parallel_analysis(Yc, 300, np.random.default_rng(2))

    Ycen = Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(Ycen, full_matrices=False)
    pc = U[:, :2] * s[:2]
    var_exp = s ** 2 / np.sum(s ** 2)
    clade_axis = Y[clade == 0].mean(0) - Y[clade == 1].mean(0)
    clade_axis /= np.linalg.norm(clade_axis)
    cos_load = abs(Vt[0] @ clade_axis)
    corr_score = abs(np.corrcoef(pc[:, 0], clade)[0, 1])

    print(f"\n=== semi-synthetic on real tree {tumor} ({n} leaves) ===")
    print(f"genes EVOLUTIONARILY INDEPENDENT (no true factor); deep split "
          f"{int((clade==0).sum())}/{int((clade==1).sum())}")
    print(f"naive: {sig_n} 'significant' factor(s), PC1={var_exp[0]*100:.0f}% var; "
          f"|corr(PC1,clade)|={corr_score:.2f}, |cos(load,clade axis)|={cos_load:.2f}")
    print(f"phylo (contrasts): {sig_p} significant factor(s)")

    # ---- repeated histories -------------------------------------------------
    fp_n = np.zeros(R, int)
    fp_p = np.zeros(R, int)
    rng2 = np.random.default_rng(100)
    for r in range(R):
        Yr = simulate_independent_genes(L_C, p, rng2)
        _, _, a = parallel_analysis(Yr, 120, np.random.default_rng(r))
        _, _, b = parallel_analysis(phylo_contrasts(Yr, C), 120, np.random.default_rng(9000 + r))
        fp_n[r], fp_p[r] = a, b
    rate_n, rate_p = float(np.mean(fp_n > 0)), float(np.mean(fp_p > 0))
    print(f"over {R} histories: P(naive>=1 factor)={rate_n:.2f}, "
          f"P(phylo>=1 factor)={rate_p:.2f} (nominal 0.05); "
          f"mean# naive={fp_n.mean():.2f} phylo={fp_p.mean():.2f}")

    # ---- figure -------------------------------------------------------------
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    for cl, col, lab in [(0, "#2c7fb8", "deep clade A"), (1, "#d95f0e", "deep clade B")]:
        m = clade == cl
        ax[0].scatter(pc[m, 0], pc[m, 1], s=16, alpha=0.7, color=col, label=lab)
    ax[0].set_xlabel(f"naive PC1 ({var_exp[0]*100:.0f}%)")
    ax[0].set_ylabel(f"naive PC2 ({var_exp[1]*100:.0f}%)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"(A) Naive FA, independent genes on {tumor}\n"
                    "PC1 tracks the deep subclonal split", fontsize=10)

    idx = np.arange(1, len(obs_n) + 1)
    ax[1].plot(idx, obs_n, "o-", color="#d95f0e", label="naive: observed")
    ax[1].plot(idx, null_n, "s--", color="#d95f0e", alpha=0.45, label="naive: null 95%")
    ax[1].plot(idx, obs_p, "o-", color="#2c7fb8", label="phylo: observed")
    ax[1].plot(idx, null_p, "s--", color="#2c7fb8", alpha=0.45, label="phylo: null 95%")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("component")
    ax[1].set_ylabel("eigenvalue")
    ax[1].legend(fontsize=8)
    ax[1].set_title("(B) Parallel analysis: naive invents a factor,\ncontrasts do not",
                    fontsize=10)

    bars = ax[2].bar([0, 1], [rate_n, rate_p], color=["#d95f0e", "#2c7fb8"], width=0.6)
    ax[2].axhline(0.05, ls=":", color="grey")
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["naive", "phylo\n(contrasts)"])
    ax[2].set_ylabel(f"P(\u22651 'significant' factor), {R} histories")
    ax[2].set_ylim(0, 1.05)
    for b, v in zip(bars, [rate_n, rate_p]):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    ax[2].set_title("(C) Illusory significance on the real tree", fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIGDIR, f"semisynth_{tumor}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return dict(rate_naive=rate_n, rate_phylo=rate_p, n=n)


if __name__ == "__main__":
    ad = anndata.read_h5ad(DATA + "/expression/adata_processed.nt.h5ad")
    for tumor in ["3726_NT_T2", "3513_NT_T3"]:
        run(tumor=tumor, adata=ad)
