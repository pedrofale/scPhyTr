"""SCOUT's clear failure mode: real events that are invisible to any PER-GENE test.

SCOUT decides gene by gene. An adaptive event is reported only if some individual gene beats AICc on
its own. That is a hard information threshold: an event whose per-gene effect is small is invisible,
no matter how many genes carry it and no matter how certain it is in aggregate.

This experiment is deliberately maximally generous to SCOUT:
  * it is GIVEN the true adaptive landscape (the thing it cannot infer);
  * data are simulated from exactly the model it fits (OU with a regime shift);
  * preprocessing is its own (log1p + lineage smoothing, k=8).
The only thing withheld is the ability to pool evidence across genes.

We sweep the per-gene effect size and compare:
  * PER-GENE (SCOUT's rule)  -- how many genes are individually called OUX;
  * POOLED BRANCH SCAN       -- sum the per-gene dAICc(OUX vs best of BM1/OU1) over genes for EVERY
                                candidate branch, and ask whether the TRUE branch wins. This is a
                                crude stand-in for P(z_b = 1 | Y): no per-gene call is required.

The gap between the two curves is the headroom the hierarchical model is designed to exploit.

    python -m analysis.scout.experiments.05_pooling_gap
"""
import os

import numpy as np
import pandas as pd

from scphytr.modes._tree import Tree
from scphytr.modes.simulate import simulate_tips, leaf_regimes
from analysis.scout.scout_preprocess import scout_preprocess
from scphytr.modes.baseline import paint_regimes, regime_design, _profile, _aicc
from scphytr.modes._ou import tip_cov

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, SIGMA, DEPTH = 3.0, 0.75, 30.0
N_TIPS, N_GENES, FRAC = 256, 400, 0.25
ALPHA_GRID = np.exp(np.linspace(np.log(0.3), np.log(12.0), 10))


def candidate_branches(tree, min_leaves=8):
    sets = tree.leaf_sets()
    return [v for v in range(tree.n_nodes)
            if tree.parent[v] >= 0 and min_leaves <= len(sets[v]) <= tree.n_leaves - min_leaves]


def simulate(tree, event_branch, shift, seed=0):
    """All genes OU; a fraction FRAC of them shift their optimum on ``event_branch``."""
    rng = np.random.default_rng(seed)
    delta = np.zeros((tree.n_nodes, N_GENES))
    responders = rng.choice(N_GENES, size=int(FRAC * N_GENES), replace=False)
    delta[event_branch, responders] = rng.normal(0.0, shift, size=len(responders))
    lat = simulate_tips(tree, ALPHA, SIGMA, rng.normal(size=N_GENES), delta, rng=rng,
                        root="stationary")
    E = np.exp(lat - lat.mean(0))
    counts = rng.poisson(DEPTH * E / E.mean(0)).astype(float)
    return counts, responders


def branch_scan(Z, tree, branches, alpha_grid=ALPHA_GRID):
    """dAICc(OUX vs best of BM1/OU1) per gene, for every candidate branch. Returns (n_branches, G).

    R(alpha) does not depend on the branch, so each alpha is factorised ONCE and reused across all
    candidate branches -- which is what makes a whole-tree scan affordable.
    """
    n, G = Z.shape
    # null models (BM1, OU1): both have a constant mean, so W is a column of ones
    R_bm = tip_cov(tree, 1e-12, 1.0, root="fixed")
    ll_bm, _, _ = _profile(Z, R_bm, np.ones((n, 1)))
    aic_bm = _aicc(ll_bm, 2, n)
    aic_ou1 = np.full(G, np.inf)
    cache = {}
    for a in alpha_grid:
        R = tip_cov(tree, a, 1.0, root="stationary")
        cache[a] = R
        ll, _, _ = _profile(Z, R, np.ones((n, 1)))
        aic_ou1 = np.minimum(aic_ou1, _aicc(ll, 3, n))
    base = np.minimum(aic_bm, aic_ou1)

    out = np.full((len(branches), G), -np.inf)
    for bi, b in enumerate(branches):
        z = np.zeros(tree.n_nodes, bool); z[b] = True
        node_regime, uniq = paint_regimes(tree, leaf_regimes(tree, z))
        if len(uniq) < 2:
            continue
        best = np.full(G, np.inf)
        for a in alpha_grid:
            W = regime_design(tree, a, node_regime, len(uniq))
            ll, _, _ = _profile(Z, cache[a], W)
            best = np.minimum(best, _aicc(ll, 2 + len(uniq), n))
        out[bi] = base - best
    return out


def main():
    tree = Tree.balanced(N_TIPS, 1.0).scale_height(1.0)
    branches = candidate_branches(tree)
    sets = tree.leaf_sets()
    true_b = next(v for v in branches if len(sets[v]) == 64)
    print(f"{tree.n_leaves} tips, {N_GENES} genes, {int(FRAC*N_GENES)} responders, "
          f"true event branch {true_b} (|clade|={len(sets[true_b])}), "
          f"{len(branches)} candidate branches\n")

    rows = []
    for shift in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
        counts, responders = simulate(tree, true_b, shift, seed=0)
        X, t = scout_preprocess(counts, tree, normalize=True, smoothing_k=8)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        gains = branch_scan(Z, t, branches)

        gi = branches.index(true_b)
        is_resp = np.zeros(N_GENES, bool); is_resp[responders] = True
        called = gains[gi] >= 2.0                       # SCOUT's stringency: dAICc > 2
        tp = float(called[is_resp].mean())              # responders correctly found
        fp = float(called[~is_resp].mean())             # non-responders wrongly called
        excess = tp - fp                                # detection above the false-positive floor
        pooled = gains.sum(axis=1)
        winner = branches[int(np.argmax(pooled))]
        rank_true = int(np.argsort(-pooled).tolist().index(gi)) + 1
        margin = float(np.sort(pooled)[-1] - np.sort(pooled)[-2])
        # ORACLE: pool only over the genes that really respond (perfect gamma) -- the ceiling that
        # a model inferring gene participation could reach
        oracle = gains[:, is_resp].sum(axis=1)
        rank_oracle = int(np.argsort(-oracle).tolist().index(gi)) + 1
        rows.append(dict(shift=shift, tp_rate=tp, fp_rate=fp, excess=excess,
                         oracle_rank=rank_oracle,
                         pooled_winner_is_true=bool(winner == true_b),
                         true_branch_rank=rank_true, pooled_margin=margin))
        print(f"shift={shift:4.2f}  PER-GENE: responders found {tp:5.1%}, false positives {fp:5.1%} "
              f"(excess {excess:+5.1%})   |   POOLED: picks "
              f"{'TRUE branch' if winner == true_b else f'branch {winner}':<12s} "
              f"rank {rank_true}/{len(branches)}   |   ORACLE-gamma rank {rank_oracle}/{len(branches)}",
              flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", "pooling_gap.csv")
    df.to_csv(out, index=False)
    print(f"\n[wrote {out}]")

    print("\nHONEST READ-OUT")
    print("  * Naive pooling (summing dAICc over ALL genes) is NOT a rescue: at small effects it")
    print("    ranks the true branch WORSE than per-gene detection would suggest, because the")
    print("    non-responding genes swamp the responders.")
    print("  * Pooling over the TRUE responder set (oracle gamma) IS a rescue: it moves the true")
    print("    branch from a middling rank to the top of the scan at effect sizes where per-gene")
    print("    detection is weak. So the leverage comes from inferring WHICH GENES RESPOND")
    print("    (gamma_bg), not merely from sharing event locations across genes.")
    print("  * That is a design constraint on Model 1, and a falsifiable target: it must approach")
    print("    the oracle curve without being told the responder set.")
    figure(df, os.path.join(HERE, "results", "figures", "pooling_gap.png"))


def figure(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sh = df["shift"].values
    ax[0].plot(sh, df.tp_rate, "-o", color="#2c7fb8", label="responders found")
    ax[0].plot(sh, df.fp_rate, "-o", color="#bbbbbb", label="false positives")
    ax[0].plot(sh, df.excess, "-o", color="#d62728", label="excess (real detection)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("per-gene effect size (optimum shift)")
    ax[0].set_ylabel("rate")
    ax[0].set_title("PER-GENE rule (SCOUT, dAICc>=2):\nreal detection vanishes as effects shrink")
    ax[0].legend(fontsize=8)

    ok = df.pooled_winner_is_true.values
    ax[1].plot(sh, df.true_branch_rank, "-o", color="#d62728", label="pool ALL genes (naive)")
    ax[1].plot(sh, df.oracle_rank, "-o", color="#2c7fb8", label="pool TRUE responders (oracle gamma)")
    ax[1].axhline(1, ls=":", color="grey", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].invert_yaxis(); ax[1].set_xlabel("per-gene effect size (optimum shift)")
    ax[1].set_ylabel("rank of the TRUE branch (1 = best)")
    ax[1].set_title("POOLED branch scan:\nonly sparsity-aware pooling localises the event")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
