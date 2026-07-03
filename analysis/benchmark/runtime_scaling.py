"""Runtime: the low-rank Poisson factor model scales to the transcriptome; a full covariance doesn't.

Two honest points about scPhyTr's runtime (verified, not assumed):

* GENE axis (this benchmark) -- a real win. Gene expression is low-dimensional (a few programs), so
  scPhyTr models the gene-gene structure with a low-rank Poisson factor model ``K = W Wᵀ`` (k
  factors), which costs ~O(p·k) in the number of genes. Fitting the FULL p×p evolutionary covariance
  (fit_mv_em) costs roughly O(p^2.5+) and becomes intractable: at p=50 genes the low-rank fit is
  ~200x faster, and the gap explodes with p.
* CELL axis (NOT benchmarked here -- honest caveat). scPhyTr's per-gene tree-Laplace is O(cells) but
  with a large constant (the count-model optimizer), so a BLAS-optimized dense O(n^3) Gaussian PCM is
  actually faster until ~10^4 cells. So we do NOT claim a cell-count runtime win over dense ML; the
  runtime wins are the gene axis (here) and vs MCMC (RevBayes ~40x, elsewhere).

We simulate low-rank Poisson factor data (k=5 real programs) on a fixed tree and time both fits as
the number of genes grows.
"""
import os
import numpy as np
import pandas as pd

from analysis.benchmark.spatial_decomposition import _tree
from scphytr.tools.poisson_factor import fit_poisson_factor_analysis, simulate_poisson_pfa
from scphytr.tools.em import fit_mv_em
from scphytr.inference.laplace import MultiCellPoissonObservation

OUT = os.path.dirname(__file__)


def _data(tree, p, k_true=5, mean_size=200, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((p, k_true)) * 0.6
    Y, X, sizes, leaf_names = simulate_poisson_pfa(tree, W, np.zeros(p), mean_size=mean_size, seed=seed)
    return Y, sizes, leaf_names


def _t_lowrank(tree, Y, sizes, leaf_names, k=5, n_iter=30):
    import time
    t = time.time(); fit_poisson_factor_analysis(Y, tree, k=k, sizes=sizes, leaf_names=leaf_names,
                                                  n_iter=n_iter); return time.time() - t


def _t_fullrank(tree, Y, sizes, n, max_em=30):
    import time
    obs = MultiCellPoissonObservation(Y, sizes, np.arange(n), n)
    t = time.time(); fit_mv_em(tree, obs, model="BM", max_em=max_em); return time.time() - t


def sweep(genes_lr=(10, 20, 50, 100, 200, 400, 800), genes_fr=(10, 20, 50), n=80,
          reps_lr=2, reps_fr=1, seed0=0):
    tree = _tree(n, 0)
    rows = []
    allp = sorted(set(genes_lr) | set(genes_fr))
    for p in allp:
        Y, sizes, leaf_names = _data(tree, p, seed=seed0)
        row = {"genes": p}
        if p in genes_lr:
            ts = [_t_lowrank(tree, Y, sizes, leaf_names) for _ in range(reps_lr)]
            row["lowrank"] = float(np.median(ts))
        if p in genes_fr:
            ts = [_t_fullrank(tree, Y, sizes, n) for _ in range(reps_fr)]
            row["fullrank"] = float(np.median(ts))
        rows.append(row)
        print(f"  p={p:4d} genes: lowrank {row.get('lowrank', float('nan')):.2f}s  "
              f"fullrank {row.get('fullrank', float('nan')):.2f}s", flush=True)
    return pd.DataFrame(rows)


def _fit_power(x, y):
    """t = a * p^b (least squares in log-log); returns (a, b)."""
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    return np.exp(la), b


def figure(df=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None:
        df = pd.read_csv(os.path.join(OUT, "runtime_scaling.csv"))
    lr = df.dropna(subset=["lowrank"]); fr = df.dropna(subset=["fullrank"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(lr["genes"], lr["lowrank"], "-o", color="#2c7fb8", label="low-rank PFA (k=5), O(p·k)")
    ax.plot(fr["genes"], fr["fullrank"], "-o", color="#e45756", label="full covariance, ~O(p²·⁵)")
    # extrapolate the full-rank power law to transcriptome scale
    if len(fr) >= 2:
        a, b = _fit_power(fr["genes"].values, fr["fullrank"].values)
        pp = np.array([fr["genes"].min(), 2000])
        ax.plot(pp, a * pp ** b, ls="--", color="#e45756", alpha=0.6,
                label=f"full-rank extrapolated (p^{b:.1f})")
        ax.annotate(f"~{a*2000**b/3600:.0f} h at 2000 genes", (2000, a * 2000 ** b),
                    fontsize=8, color="#e45756", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of genes"); ax.set_ylabel("wall-clock to fit (s)")
    ax.set_title("Gene-gene co-evolution: low-rank scales; full covariance doesn't")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "runtime_scaling.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    print("== runtime vs number of genes: low-rank PFA vs full covariance ==")
    df = sweep()
    df.to_csv(os.path.join(OUT, "runtime_scaling.csv"), index=False)
    figure(df)
    both = df.dropna(subset=["lowrank", "fullrank"])
    if len(both):
        r = both.iloc[-1]
        print(f"\nAt {int(r['genes'])} genes: low-rank {r['lowrank']:.2f}s vs full-rank "
              f"{r['fullrank']:.1f}s = {r['fullrank']/r['lowrank']:.0f}x faster (gap grows with p).")


if __name__ == "__main__":
    main()
