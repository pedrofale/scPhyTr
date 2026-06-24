"""OU-2 adaptive-expression scan on the melanoma subclone counts (dispersion-aware).

Per gene, fit BM / OU-1 / OU-2 to the multi-cell Poisson observation (cells within
each subclone leaf) and select by AIC. OU-2 = the paper's adaptive model (a
"chosen" subline group shifts to a different optimum). Within-subclone plasticity
is folded in: a per-gene NB dispersion ``r`` is estimated once (NB MLE at the
empirical per-subclone means) and held fixed across the three models, so the
comparison is about the *between*-subclone optimum shift, not the within-clone
noise. A gene selected OU-2 is adaptively shifted in the chosen sublines.

Usage: python -m analysis.melanoma.adaptive_ou2 [regime] [n_genes] [poisson|nb]
"""
import sys
import numpy as np
import pandas as pd

from analysis.melanoma.load import (
    load_counts, tree_leaves, load_tree, load_regimes,
)
from scphytr.inference.laplace import MultiCellPoissonObservation
from scphytr.tools.model_selection import (
    fit_bm_counts, fit_ou_counts, fit_ou_regimes_counts,
)


def select_genes(X, genes, n_genes, all_sublines_idx):
    """Top-``n_genes`` highly-variable genes expressed in every subline."""
    leaf_of, idx = all_sublines_idx
    nL = len(set(idx))
    # per-subline mean log1p, keep genes detected in all sublines
    import pandas as pd
    df = pd.DataFrame(np.log1p(X)); df["__c"] = idx
    pb = df.groupby("__c").mean()
    expressed = ((X > 0).astype(float))
    det = pd.DataFrame(expressed); det["__c"] = idx
    detrate = det.groupby("__c").mean().min(axis=0)        # min over sublines
    ok = np.where(detrate.values > 0)[0]
    var = pb.iloc[:, ok].var(axis=0).values
    pick = ok[np.argsort(var)[::-1][:n_genes]]
    return list(pick)


def scan(regime="har", n_genes=20, dispersion="nb", verbose=True, gene_symbols=None):
    X, genes, clone, sf = load_counts()
    leaves = tree_leaves()
    leaf_of = {name: k for k, name in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone])
    nL = len(leaves)

    if gene_symbols is not None:
        # explicit gene list (symbols); keep those present and detected in all sublines
        sym = np.array([g.split("_")[-1] for g in genes])
        det = pd.DataFrame((X > 0).astype(float)); det["__c"] = idx
        detmin = det.groupby("__c").mean().min(axis=0).values
        want = set(gene_symbols)
        pick = [i for i in range(len(genes)) if sym[i] in want and detmin[i] > 0]
    else:
        pick = select_genes(X, genes, n_genes, (leaf_of, idx))
    names = [genes[g].split("_")[-1] for g in pick]

    tree = load_tree()
    regimes, n_reg = load_regimes(tree, regime)
    if verbose:
        chosen = sorted(nd.name for nd in tree.phylotree.get_leaves() if regimes[nd] == 1)
        print(f"regime '{regime}': chosen sublines = {chosen}; "
              f"scanning {len(pick)} genes ({dispersion})\n")

    rows = []
    for g, nm in zip(pick, names):
        Xg = X[:, [g]]
        if dispersion == "nb":
            obs0 = MultiCellPoissonObservation(Xg, sf, idx, nL, dispersion=10.0,
                                               univariate=True)
            r = float(obs0.update_dispersion(obs0.mode_init())[0])
            obs = MultiCellPoissonObservation(Xg, sf, idx, nL, dispersion=r,
                                              univariate=True)
            kdisp = 1
        else:
            obs = MultiCellPoissonObservation(Xg, sf, idx, nL, dispersion=None,
                                              univariate=True)
            r, kdisp = np.inf, 0

        # Fast scalar tree-Laplace path (no 1x1 LAPACK overhead of the MV path).
        rbm = fit_bm_counts(tree, obs)
        rou = fit_ou_counts(tree, obs)
        rou2 = fit_ou_regimes_counts(tree, obs, regimes, n_reg)
        ll = {"BM": rbm.loglik, "OU1": rou.loglik, "OU2": rou2.loglik}
        k = {"BM": rbm.n_params + kdisp, "OU1": rou.n_params + kdisp,
             "OU2": rou2.n_params + kdisp}
        aic = {m: 2 * k[m] - 2 * ll[m] for m in ll}
        sel = min(aic, key=aic.get)
        runner = sorted(aic.values())
        rows.append({
            "gene": nm, "selected": sel, "adaptive": sel == "OU2",
            "d_aic": runner[1] - runner[0], "disp_r": r,
            "aic_BM": aic["BM"], "aic_OU1": aic["OU1"], "aic_OU2": aic["OU2"],
        })
        if verbose:
            flag = "  <-- adaptive" if sel == "OU2" else ""
            print(f"{nm:>12} BM={aic['BM']:.1f} OU1={aic['OU1']:.1f} "
                  f"OU2={aic['OU2']:.1f}  -> {sel}{flag}")

    df = pd.DataFrame(rows).set_index("gene")
    if verbose:
        n_ad = int(df["adaptive"].sum())
        print(f"\n{n_ad}/{len(df)} genes adaptive (OU-2) in regime '{regime}'.")
    return df


if __name__ == "__main__":
    import os
    reg = sys.argv[1] if len(sys.argv) > 1 else "har"
    ng = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    disp = sys.argv[3] if len(sys.argv) > 3 else "nb"
    df = scan(reg, ng, disp)
    out = os.path.join(os.path.dirname(__file__), "figures", f"adaptive_{reg}_{disp}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out)
    print(f"saved {out}")
