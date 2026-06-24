"""Ground-truth benchmark: heritability detection vs sequencing depth.

Common task all three tools address: does a gene's expression carry phylogenetic
signal (heritable, structured on the tree) or is it plastic noise (iid across
cells)? Ground truth: "signal" genes have a BM latent on the tree; "noise" genes
have iid per-subline latents. Cells are Poisson draws; we sweep depth.

Scored threshold-free by AUROC (separating signal from noise):
  * scPhyTr  : BM-on-tree vs BM-on-star marginal gap, per-cell Poisson counts.
  * EvoGeneX : same gap, Gaussian on per-subline mean log1p(count).
  * PATH     : Moran's I phylogenetic autocorrelation of per-subline mean log1p.

scPhyTr uses counts; EvoGeneX and PATH use the normalized trait -- so the latter
two are expected to lose AUROC as depth drops, while the count model holds up.
"""
import numpy as np
from ete3 import PhyloTree

from analysis.melanoma.load import tree_leaves, load_tree
from analysis.benchmark.path_morans import phylo_weights, morans_I
from scphytr.utils.tree import Tree
from scphytr.inference.laplace import MultiCellPoissonObservation
from scphytr.tools.model_selection import fit_bm_counts, fit_bm


def _wrap(nwk):
    t = Tree(); t.phylotree = PhyloTree(nwk, format=1)
    t.root = t.phylotree.get_tree_root()
    return t


def star_tree(leaves, height):
    return _wrap("(" + ",".join(f"{l}:{height:.6f}" for l in leaves) + ");")


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum(); n0 = (~labels.astype(bool)).sum()
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sim_subline_rate(tree, leaves, base, tip_std, signal, rng):
    """Per-subline latent log-rate: BM on tree (signal) or iid (noise)."""
    H = float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)
    if signal:
        dev = {}
        for nd in tree.phylotree.traverse("preorder"):
            dev[nd] = 0.0 if nd.up is None else dev[nd.up] + rng.normal(0, tip_std * np.sqrt(nd.dist / H))
        z = np.array([dev[l] for l in tree.phylotree.get_leaves()])
    else:
        z = rng.normal(0, tip_std, size=len(leaves))   # iid: no phylogenetic signal
    return np.log(base) + z


def main(M=20, depths=(2.0, 10.0, 50.0), tip_std=0.4, cells=6, seed=0):
    rng = np.random.default_rng(seed)
    tree = load_tree(); leaves = tree_leaves(); nL = len(leaves)
    H = float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)
    star = star_tree(leaves, H)
    W = phylo_weights(tree, leaves)
    idx = np.repeat(np.arange(nL), cells); sf = np.ones(idx.shape[0])

    print(f"heritability detection: {M} signal + {M} noise genes/depth "
          f"({cells} cells/subline). AUROC (signal vs noise):\n")
    print(f"{'depth':>6} | {'scPhyTr':>8} {'EvoGeneX':>9} {'PATH':>6}")
    for d in depths:
        sc, eg, pa, lab = [], [], [], []
        for signal in (True, False):
            for _ in range(M):
                z = sim_subline_rate(tree, leaves, d, tip_std, signal, rng)
                y = rng.poisson(sf * np.exp(z[idx])).astype(float)
                # scPhyTr: BM-tree vs BM-star marginal gap (counts)
                obs = MultiCellPoissonObservation(y[:, None], sf, idx, nL, univariate=True)
                sc.append(fit_bm_counts(tree, obs).loglik - fit_bm_counts(star, obs).loglik)
                # EvoGeneX + PATH: per-subline mean log1p
                sub = np.array([np.mean(np.log1p(y[idx == k])) for k in range(nL)])
                vals = {leaves[k]: float(sub[k]) for k in range(nL)}
                vals_star = dict(vals)
                eg.append(fit_bm(tree, vals).loglik - fit_bm(star, vals_star).loglik)
                pa.append(morans_I(sub, W))
                lab.append(1 if signal else 0)
        lab = np.array(lab)
        print(f"{d:6.0f} | {auroc(sc, lab):8.2f} {auroc(eg, lab):9.2f} {auroc(pa, lab):6.2f}")


if __name__ == "__main__":
    main()
