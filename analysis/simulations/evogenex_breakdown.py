"""Where Gaussian-on-log (EvoGeneX-style) breaks and the count model (scPhyTr) works.

We simulate a single-gene adaptive-expression test on a real subline tree:
a "chosen" clade is silenced (down-shift Delta in latent log-expression) under the
alternative, or not (null). Cells per subline are Poisson draws around the
subline's latent rate. We sweep the *sequencing depth* (baseline mean counts per
cell) and, at each depth, measure each method's power (alt called adaptive) and
false-positive rate (null called adaptive) for the BM-vs-OU2 test.

Both methods use the *same* tree and the *same* BM / OU-2 models -- only the
observation model differs:
  * scPhyTr   : per-cell Poisson counts (multi-cell observation), latent tree-Laplace.
  * EvoGeneX  : per-subline mean of log1p(count) (log-normalized pseudobulk),
                Gaussian phylogenetic BM/OU (the Gaussian-on-log approach).
So any difference is attributable to the observation model, which is the point:
at low depth the log transform + shot noise wash out the shift for Gaussian-on-log,
while the Poisson model still resolves it.
"""
import numpy as np

from analysis.melanoma.load import tree_leaves, load_tree, load_regimes
from scphytr.inference.laplace import MultiCellPoissonObservation
from scphytr.tools.model_selection import (
    fit_bm_counts, fit_ou_regimes_counts, fit_bm, fit_ou_regimes,
)


def simulate_gene(tree, leaves, idx, sf, chosen_mask, base_count, sigma_bm,
                  delta, alt, rng):
    """Latent BM log-rate per subline (+ down-shift on chosen if alt); Poisson cells."""
    # BM drift down the tree -> per-subline latent deviation
    dev = {}
    for nd in tree.phylotree.traverse("preorder"):
        if nd.up is None:
            dev[nd] = 0.0
        else:
            dev[nd] = dev[nd.up] + rng.normal(0, np.sqrt(sigma_bm * nd.dist))
    z0 = np.log(base_count)
    z = np.array([z0 + dev[l] for l in tree.phylotree.get_leaves()])   # leaf order
    if alt:
        z = z - delta * chosen_mask                                    # silence chosen
    rate = sf * np.exp(z[idx])                                         # per cell
    return rng.poisson(rate).astype(float)


def adaptive_scphytr(tree, y, sf, idx, nL, regimes, n_reg):
    obs = MultiCellPoissonObservation(y[:, None], sf, idx, nL, dispersion=None,
                                      univariate=True)
    bm = fit_bm_counts(tree, obs)
    ou2 = fit_ou_regimes_counts(tree, obs, regimes, n_reg)
    return ou2.aic() < bm.aic()


def adaptive_evogenex(tree, y, idx, leaves, regimes, n_reg):
    # EvoGeneX surrogate: per-subline mean of log1p(count) as the Gaussian trait
    vals = {}
    for k, name in enumerate(leaves):
        cells = y[idx == k]
        vals[name] = float(np.mean(np.log1p(cells)))
    bm = fit_bm(tree, vals)
    ou2 = fit_ou_regimes(tree, vals, regimes, n_reg)
    return ou2.aic() < bm.aic()


def main(M=40, depths=(2.0, 8.0, 40.0), delta=1.5, tip_std=0.25,
         cells_per_subline=6, seed=0):
    rng = np.random.default_rng(seed)
    tree = load_tree()
    leaves = tree_leaves()
    regimes, n_reg = load_regimes(tree, "har")
    chosen_mask = np.array([1.0 if regimes[l] == 1 else 0.0
                            for l in tree.phylotree.get_leaves()])
    nL = len(leaves)
    # neutral BM rate set so root-to-tip drift std ~ tip_std (log units)
    H = float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)
    sigma_bm = tip_std ** 2 / H
    idx = np.repeat(np.arange(nL), cells_per_subline)
    sf = np.ones(idx.shape[0])

    print(f"adaptive-detection: {M} null + {M} alt genes/depth, delta={delta} "
          f"(chosen sublines silenced), {cells_per_subline} cells/subline\n")
    print(f"{'depth':>6} | {'scPhyTr power':>13} {'FPR':>5} | {'EvoGeneX power':>14} {'FPR':>5}")
    rows = []
    for d in depths:
        tp = {"sc": 0, "eg": 0}; fp = {"sc": 0, "eg": 0}
        for _ in range(M):
            ya = simulate_gene(tree, leaves, idx, sf, chosen_mask, d, sigma_bm,
                               delta, True, rng)
            yn = simulate_gene(tree, leaves, idx, sf, chosen_mask, d, sigma_bm,
                               delta, False, rng)
            tp["sc"] += adaptive_scphytr(tree, ya, sf, idx, nL, regimes, n_reg)
            tp["eg"] += adaptive_evogenex(tree, ya, idx, leaves, regimes, n_reg)
            fp["sc"] += adaptive_scphytr(tree, yn, sf, idx, nL, regimes, n_reg)
            fp["eg"] += adaptive_evogenex(tree, yn, idx, leaves, regimes, n_reg)
        row = dict(depth=d, sc_power=tp["sc"]/M, sc_fpr=fp["sc"]/M,
                   eg_power=tp["eg"]/M, eg_fpr=fp["eg"]/M)
        rows.append(row)
        print(f"{d:6.0f} | {row['sc_power']:13.2f} {row['sc_fpr']:5.2f} | "
              f"{row['eg_power']:14.2f} {row['eg_fpr']:5.2f}")
    return rows


if __name__ == "__main__":
    main()
