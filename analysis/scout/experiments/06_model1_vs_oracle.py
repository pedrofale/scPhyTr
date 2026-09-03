"""Model 1 vs SCOUT on the pooling gap: can the posterior reach the oracle without the oracle?

Experiment 05 established the target. Pooling per-gene dAICc over ALL genes ranks the true event
branch badly (non-responders swamp responders), while pooling over only the TRUE responders --
"oracle gamma", which requires knowing the answer -- ranks it near the top. So the leverage is in
inferring gamma_bg, and the falsifiable claim for Model 1 is:

    approach the oracle curve WITHOUT being told the responder set.

Everything is maximally generous to SCOUT: it is given the true adaptive landscape, its own
preprocessing (log1p + lineage smoothing, k=8), and data simulated from an OU regime shift.
Model 1 is given the tree and the expression matrix only -- no landscape, no responders, no alpha.

    python -m analysis.scout.experiments.06_model1_vs_oracle
"""
import importlib.util
import os

import numpy as np
import pandas as pd

from scphytr.modes._tree import Tree
from analysis.scout.scout_preprocess import scout_preprocess
from scphytr.modes.model import fit_model1, branch_evidence

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "exp05", os.path.join(HERE, "experiments", "05_pooling_gap.py"))
exp05 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp05)

ALPHA_GRID = np.exp(np.linspace(np.log(0.5), np.log(8.0), 8))
TAU_GRID = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
SHIFTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def auroc(score, label):
    o = np.argsort(score)
    ranks = np.empty(len(score)); ranks[o] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    tree = Tree.balanced(exp05.N_TIPS, 1.0).scale_height(1.0)
    branches = exp05.candidate_branches(tree)
    sets = tree.leaf_sets()
    true_b = next(v for v in branches if len(sets[v]) == 64)
    gi = branches.index(true_b)
    G = exp05.N_GENES
    print(f"{tree.n_leaves} tips, {G} genes, {int(exp05.FRAC*G)} responders, true branch {true_b} "
          f"(|clade|={len(sets[true_b])}), {len(branches)} candidates\n")

    rows = []
    for shift in SHIFTS:
        counts, responders = exp05.simulate(tree, true_b, shift, seed=0)
        X, t = scout_preprocess(counts, tree, normalize=True, smoothing_k=8)   # SCOUT's best arm
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Xr, _ = scout_preprocess(counts, tree, normalize=True, smoothing_k=None)  # log-norm only
        Zr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9)
        is_resp = np.zeros(G, bool); is_resp[responders] = True

        # ---- SCOUT, given the true landscape, and the two pooled scans (experiment 05) ----
        gains = exp05.branch_scan(Z, t, branches)
        called = gains[gi] >= 2.0
        pooled = gains.sum(axis=1)
        rank_pool = int(np.argsort(-pooled).tolist().index(gi)) + 1
        orc = gains[:, is_resp].sum(axis=1)
        rank_orc = int(np.argsort(-orc).tolist().index(gi)) + 1

        # ---- Model 1's collapsed branch evidence (no MCMC, no responder set) ----
        _, ev = branch_evidence(Zr, t, branches=np.array(branches), alpha_grid=ALPHA_GRID,
                                tau_grid=TAU_GRID, omega=1.0, rho=0.1, standardize=False)
        rank_ev = int(np.argsort(-ev).tolist().index(gi)) + 1

        # ---- Model 1: tree + expression only ----
        res = fit_model1(Zr, t, branches=np.array(branches), alpha_grid=ALPHA_GRID,
                         tau_grid=TAU_GRID, omega=1.0, n_iter=1500, burn=500, seed=1,
                         standardize=False)
        bi = list(res.branches).index(true_b)
        rank_m1 = int(np.argsort(-res.p_z).tolist().index(bi)) + 1
        pg = res.p_gamma[bi]
        rows.append(dict(shift=shift,
                         scout_tp=float(called[is_resp].mean()), scout_fp=float(called[~is_resp].mean()),
                         rank_pooled=rank_pool, rank_oracle=rank_orc,
                         rank_evidence=rank_ev, rank_model1=rank_m1,
                         p_z_true=float(res.p_z[bi]), n_events=float(res.n_event_draws.mean()),
                         gamma_auroc=auroc(pg, is_resp),
                         alpha_post=float(np.median(res.alpha_draws)),
                         tau_post=float(np.median(res.tau_draws))))
        r = rows[-1]
        print(f"shift {shift:.2f} | SCOUT/gene tp {r['scout_tp']:.2f} fp {r['scout_fp']:.2f} | "
              f"rank pooled {rank_pool:2d} oracle {rank_orc:2d} EVID {rank_ev:2d} "
              f"MODEL1 {rank_m1:2d} | "
              f"P(z)={r['p_z_true']:.2f} gammaAUROC={r['gamma_auroc']:.3f} "
              f"nev={r['n_events']:.1f} a={r['alpha_post']:.1f} tau={r['tau_post']:.2f}")

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", "model1_vs_oracle.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
