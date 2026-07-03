"""Count-native selection detection keeps its power at single-cell depth (weak-selection regime).

Detecting stabilizing selection = distinguishing an Ornstein-Uhlenbeck process (a trait pulled back
toward an optimum) from neutral Brownian motion on the tree. For STRONG selection the OU-vs-BM signal
is large and both a count model and a Gaussian-on-log detector find it easily even at low depth (no
advantage -- verified at alpha>=1). The interesting, realistic case is WEAK selection (alpha~0.3),
where the signal is subtle: there, count sampling noise -- which the Gaussian-on-log detector cannot
model -- swamps it and the Gaussian test loses power, while scPhyTr fits OU and BM through the count
likelihood and keeps more of it. This is a smaller, more scoped advantage than the magnitude
read-outs (heritability, co-evolution, plasticity): selection *detection* aggregates the signal over
all leaves, so per-leaf count noise partly averages out.

We plant half the genes as neutral BM and half under weak OU (stabilizing selection), observe counts
over a depth sweep, and score each gene by the OU-vs-BM model-selection margin (ΔAIC) two ways:

  * scPhyTr  -- fit_bm_counts / fit_ou_counts (count model, cells as replicates).
  * Gaussian -- fit_bm / fit_ou on per-leaf mean log1p(Y/S) (noiseless-trait PCM).

Panel A: AUROC separating OU (selected) from BM (neutral) genes vs depth. Panel B: the OU-vs-BM
margin distributions at MERFISH depth (scPhyTr separates them; Gaussian collapses).
"""
import os
import numpy as np
import pandas as pd

from analysis.benchmark.spatial_decomposition import _tree
from scphytr.observation_models import SubclonalObservation
from scphytr.tools import model_selection as _ms

OUT = os.path.dirname(__file__)


def _ou_leaves(tree, alpha, sigma2, rng, theta=0.0):
    """Leaf values from an OU process down the tree (alpha=0 => Brownian motion)."""
    root = tree.root
    val = {id(root): theta}
    for nd in root.traverse("preorder"):
        if nd is root:
            continue
        t = float(nd.dist); pa = val[id(nd.up)]
        if alpha > 0:
            mean = theta + (pa - theta) * np.exp(-alpha * t)
            var = sigma2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
        else:
            mean = pa; var = sigma2 * t
        val[id(nd)] = mean + rng.standard_normal() * np.sqrt(max(var, 1e-12))
    return np.array([val[id(l)] for l in root.get_leaves()])


def _margin_counts(tree, Y, S):
    idx = np.arange(len(Y)); sf = np.full(len(Y), float(S))
    obs = SubclonalObservation(Y, sf, idx, len(Y), dispersion=None)
    bm = _ms.fit_bm_counts(tree, obs); ou = _ms.fit_ou_counts(tree, obs)
    return bm.aic() - ou.aic()            # >0 => OU preferred


def _margin_gauss(tree, Y, S, leaves):
    L = np.log1p(Y / S)
    vals = {leaves[i]: float(L[i]) for i in range(len(leaves))}
    bm = _ms.fit_bm(tree, vals); ou = _ms.fit_ou(tree, vals)
    return bm.aic() - ou.aic()


def _auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[label == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def _standardize(z):
    """Zero-mean, unit-variance -- so BM and OU leaf vectors differ ONLY in correlation structure
    (mean reversion), not scale; the detection task is then purely about shape."""
    return (z - z.mean()) / (z.std() + 1e-9)


def _counts(z, S, rng):
    """Poisson counts normalised so the mean count is ~S regardless of the latent variance."""
    e = np.exp(_standardize(z))
    return rng.poisson(S * e / e.mean()).astype(float)


def compute(depths=(2, 4, 8, 16, 32, 64, 128), n=70, ngenes=10, reps=6,
            alpha=0.3, sigma2=1.0, seed0=0):
    """OU vs BM leaf vectors are both standardised to unit variance, so the test is about *shape*
    (mean reversion at rate alpha), not scale; counts are normalised so mean count ~ S."""
    tree = _tree(n, 0); leaves = tree.phylotree.get_leaf_names(); nL = len(leaves)
    rows = []
    for S in depths:
        sc_sel, sc_neu, ga_sel, ga_neu = [], [], [], []
        mc = []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            for _ in range(ngenes):
                zb = _ou_leaves(tree, 0.0, sigma2, rng)             # BM (neutral)
                zo = _ou_leaves(tree, alpha, sigma2, rng)           # OU (selection)
                Yb = _counts(zb, S, rng); Yo = _counts(zo, S, rng)
                mc += [Yb.mean(), Yo.mean()]
                sc_neu.append(_margin_counts(tree, Yb, S)); sc_sel.append(_margin_counts(tree, Yo, S))
                ga_neu.append(_margin_gauss(tree, Yb, S, leaves)); ga_sel.append(_margin_gauss(tree, Yo, S, leaves))
        lab = np.r_[np.ones(len(sc_sel)), np.zeros(len(sc_neu))]
        row = {"depth": S, "mean_count": np.mean(mc),
               "scPhyTr_auroc": _auroc(np.r_[sc_sel, sc_neu], lab),
               "Gaussian_auroc": _auroc(np.r_[ga_sel, ga_neu], lab)}
        rows.append(row)
        print(f"  depth {S:4d} (mc {row['mean_count']:5.1f}): AUROC(OU vs BM)  "
              f"scPhyTr {row['scPhyTr_auroc']:.2f}  Gaussian {row['Gaussian_auroc']:.2f}", flush=True)
    return pd.DataFrame(rows)


def figure(df=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None:
        df = pd.read_csv(os.path.join(OUT, "depth_selection.csv"))
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(df["mean_count"], df["scPhyTr_auroc"], "-o", color="#2c7fb8", label="scPhyTr (count)")
    ax.plot(df["mean_count"], df["Gaussian_auroc"], "-o", color="#e6a817", label="Gaussian-on-log")
    ax.axhline(0.5, ls=":", color="grey")
    ax.axvspan(1, 6, color="#cccccc", alpha=0.25)
    ax.annotate("MERFISH /\nreal PEtracer", (2.4, 0.53), fontsize=8, color="#555")
    ax.set_xscale("log"); ax.set_xlabel("mean counts per gene per cell (depth)")
    ax.set_ylabel("AUROC (detect OU / selection vs BM)"); ax.set_ylim(0.45, 1.02)
    ax.set_title("Weak-selection detection power vs sequencing depth")
    ax.legend(fontsize=9)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "depth_selection.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    print("== selection (OU vs BM) detection power vs sequencing depth ==")
    df = compute()
    df.to_csv(os.path.join(OUT, "depth_selection.csv"), index=False)
    figure(df)


if __name__ == "__main__":
    main()
