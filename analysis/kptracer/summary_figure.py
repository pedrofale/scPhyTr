"""Cross-tumor summary: where naive and phylo factor analysis diverge.

Runs the naive-vs-phylo comparison across several KP-Tracer tumors and shows
that the divergence is *concentrated* on the clonal-identity axis: most programs
(cell states) are shared, but phylo factor analysis consistently strips the
clonality of the single most clone-aligned program.

Outputs analysis/kptracer/figures/realcompare_summary.png.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata

from analysis.kptracer.real_data_compare import run

DATA = "data/external/KPTracer-Data"
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")

TUMORS = ["3726_NT_T2", "3430_NT_T2", "3513_NT_T3", "3434_NT_T1"]


def main():
    ad = anndata.read_h5ad(DATA + "/expression/adata_processed.nt.h5ad")
    res = [run(tumor=t, adata=ad) for t in TUMORS]

    labels = [f"{r['tumor']}\n(n={r['n']})" for r in res]
    naive_max = [r["eta_naive"].max() for r in res]
    phylo_max = [r["eta_phylo"].max() for r in res]
    sub_err = [r["sub_err"] for r in res]
    max_angle = [r["angles"].max() for r in res]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(res))

    # (A) clonality of the most clone-aligned program: naive vs phylo
    ax[0].bar(x - 0.18, naive_max, width=0.36, color="#d95f0e", label="naive")
    ax[0].bar(x + 0.18, phylo_max, width=0.36, color="#2c7fb8", label="phylo")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel(r"max clade variance fraction $\eta^2$")
    ax[0].legend(fontsize=9)
    ax[0].set_title("(A) Phylo strips the clonality of the\nmost clone-aligned program",
                    fontsize=10)

    # (B) overall loading-subspace difference and the single rotated program
    ax[1].bar(x - 0.18, sub_err, width=0.36, color="#756bb1", label="subspace error")
    ax2 = ax[1].twinx()
    ax2.plot(x + 0.18, max_angle, "o", color="#31a354", ms=9, label="max principal angle")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("loading subspace error", color="#756bb1")
    ax2.set_ylabel("max principal angle (deg)", color="#31a354")
    ax[1].set_title("(B) The difference is concentrated:\none program rotates", fontsize=10)

    # (C) per-program eta^2 for the most divergent tumor (3513)
    j = TUMORS.index("3513_NT_T3")
    en = np.sort(res[j]["eta_naive"])[::-1]
    ep = np.sort(res[j]["eta_phylo"])[::-1]
    kk = np.arange(1, len(en) + 1)
    ax[2].plot(kk, en, "o-", color="#d95f0e", label="naive")
    ax[2].plot(kk, ep, "o-", color="#2c7fb8", label="phylo")
    ax[2].set_xlabel("program (sorted by $\\eta^2$)")
    ax[2].set_ylabel(r"clade variance fraction $\eta^2$")
    ax[2].legend(fontsize=9)
    ax[2].set_title("(C) 3513_NT_T3: naive has one strongly\nclonal program; phylo does not",
                    fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "realcompare_summary.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[fig] wrote {out}")

    print("\n=== cross-tumor summary ===")
    for r in res:
        print(f"{r['tumor']}: subspace_err={r['sub_err']:.2f}, max_angle={r['angles'].max():.1f} deg, "
              f"naive max eta^2={r['eta_naive'].max():.2f}, phylo max eta^2={r['eta_phylo'].max():.2f}")


if __name__ == "__main__":
    main()
