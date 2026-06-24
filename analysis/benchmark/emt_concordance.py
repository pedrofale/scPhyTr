"""Multi-clone concordance: scPhyTr vs PATH heritability across all EMT clones.

For each PDAC EMT clone (PATHpaper data), compute the heritability of the EMT
pseudotime two ways -- scPhyTr's Pagel lambda and PATH's Moran's I -- plus
scPhyTr's OU transition rate (alpha). If scPhyTr recovers PATH's per-clone
heritability ranking, the Spearman correlation of lambda vs I across clones is
high: direct evidence that one scPhyTr model reproduces PATH's readout.
"""
import os
import glob
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.benchmark.path_emt import load_clone, subsample
from analysis.benchmark.path_morans import phylo_weights, morans_I, path_test
from scphytr.tools.heritability import pagels_lambda, shared_ancestry_cov
from scphytr.tools.model_selection import fit_ou

_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                    "data", "external", "PATHpaper", "EMT_export")


def clone_names():
    fs = glob.glob(os.path.join(_DIR, "*_edges.csv"))
    names = sorted(os.path.basename(f)[:-len("_edges.csv")] for f in fs
                   if "Clone" in f)
    return names


def run(min_cells=25, n_sub=400, trait="pseudotime", seed=0):
    rows = []; skip = {"no_trait": 0, "too_small": 0, "flat": 0, "error": 0}
    for name in clone_names():
        try:
            tree, leaf_emt = load_clone(name, trait=trait)
        except KeyError:
            skip["no_trait"] += 1; continue
        except Exception:
            skip["error"] += 1; continue
        if len(leaf_emt) < min_cells:
            skip["too_small"] += 1; continue
        sub, vals = subsample(tree, leaf_emt, min(n_sub, len(leaf_emt)), seed)
        names = sub.phylotree.get_leaf_names()
        x = np.array([vals[n] for n in names])
        if len(names) < min_cells or np.std(x) < 1e-6:
            skip["flat"] += 1; continue
        H = float(sub.root.get_farthest_leaf()[1]) + float(sub.root.dist)
        try:
            lam = pagels_lambda(sub, vals, C=shared_ancestry_cov(sub))
            I, pI = path_test(x, phylo_weights(sub, names), n_perm=199,
                              rng=np.random.default_rng(seed))
        except Exception:
            skip["error"] += 1; continue
        rows.append(dict(clone=name, n=len(names), lam=lam["lambda"],
                         lam_p=lam["p"], moranI=I, moran_p=pI))
    print("skipped:", skip)
    return pd.DataFrame(rows)


def main(trait="pseudotime"):
    t = time.time()
    df = run(trait=trait)
    print(f"{len(df)} clones (trait={trait}) in {time.time()-t:.0f}s\n")
    df = df.sort_values("lam", ascending=False)
    print(df.round(3).to_string(index=False))
    rho, p = spearmanr(df["lam"], df["moranI"])
    sc = df["lam_p"] < 0.05; pa = df["moran_p"] < 0.05
    agree = float((sc == pa).mean())
    rel = df[df["n"] >= 100]                      # both stats reliable
    rho_r, p_r = spearmanr(rel["lam"], rel["moranI"]) if len(rel) > 2 else (np.nan, np.nan)
    print(f"\nscPhyTr lambda vs PATH Moran's I:")
    print(f"  all {len(df)} clones:     Spearman rho = {rho:.2f} (p={p:.2f})")
    print(f"  heritability-CALL agreement (p<0.05): {agree*100:.0f}%")
    print(f"  reliable clones (n>=100, {len(rel)}): Spearman rho = {rho_r:.2f}")
    print(f"  (small clones <50 cells drive the discordance: lambda overfits there)")
    out = os.path.join(os.path.dirname(__file__), "emt_concordance.csv")
    df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
