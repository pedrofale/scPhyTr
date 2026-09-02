"""Supplying the WRONG adaptive landscape makes per-gene OU selection report spurious adaptation.

SCOUT-style methods test BM1 vs OU1 vs OUx where the OUx regime partition is **supplied by the
user**. Model selection then conditions on that partition being correct and returns no evidence
about it. This asks what happens when it is wrong.

Design. Genes come in two groups on one tree:
  * NULL genes   -- no adaptive event anywhere (BM1 and OU1). Any OUx call on these is SPURIOUS.
  * SIGNAL genes -- a real optimum shift on two true event branches.
We then classify everything under a range of supplied landscapes:
  true / parent-of-true / sibling-of-true / child-of-true / a size-matched distant clade /
  shuffled leaf labels
and record how often each yields an "adaptive" call, plus the AICc evidence behind it.

The expected -- and more damning -- pattern is not that random labels create false positives (AICc
penalises the extra optima), but that *phylogenetically plausible but wrong* branches inherit the
signal and score comparably to the truth, so a confident OUx call does not localise the event.

    python -m analysis.scout.experiments.04_wrong_branches
"""
import os

import numpy as np
import pandas as pd

from scphytr.modes._tree import Tree
from scphytr.modes.simulate import simulate_tips, leaf_regimes
from analysis.scout.scout_preprocess import scout_preprocess
from scphytr.modes.baseline import classify_genes, fit_models, paint_regimes

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, SIGMA, SHIFT, DEPTH, G = 3.0, 0.75, 2.0, 30.0, 30


def _mask(tree, branches):
    z = np.zeros(tree.n_nodes, bool)
    for b in branches:
        if b is not None:
            z[b] = True
    return z


def two_clades(tree, lo=0.15, hi=0.40):
    sets = tree.leaf_sets(); n = tree.n_leaves
    cand = [v for v in range(tree.n_nodes) if tree.parent[v] >= 0 and lo * n <= len(sets[v]) <= hi * n]
    a = cand[0]; la = set(sets[a].tolist())
    return a, next(b for b in cand[1:] if not (la & set(sets[b].tolist())))


def relatives(tree, b):
    """parent, sibling and first child of branch b (each identified by its child node)."""
    p = int(tree.parent[b])
    sib = next((c for c in tree.children[p] if c != b), None)
    kid = tree.children[b][0] if tree.children[b] else None
    return p, sib, kid


def jaccard(tree, a, b):
    s = tree.leaf_sets()
    A, B = set(s[a].tolist()), set(s[b].tolist())
    return len(A & B) / max(len(A | B), 1)


def simulate(tree, b1, b2, seed=0):
    rng = np.random.default_rng(seed)
    zero = np.zeros((tree.n_nodes, G))
    y_bm = simulate_tips(tree, 1e-12, SIGMA, rng.normal(size=G), zero, rng=rng, root="fixed")
    y_ou = simulate_tips(tree, ALPHA, SIGMA, rng.normal(size=G), zero, rng=rng, root="stationary")
    d = np.zeros((tree.n_nodes, G))
    d[b1] = rng.normal(0, SHIFT, size=G); d[b2] = rng.normal(0, SHIFT, size=G)
    y_ox = simulate_tips(tree, ALPHA, SIGMA, rng.normal(size=G), d, rng=rng, root="stationary")
    lat = np.concatenate([y_bm, y_ou, y_ox], axis=1)
    grp = np.array(["NULL-BM1"] * G + ["NULL-OU1"] * G + ["SIGNAL"] * G, dtype=object)
    E = np.exp(lat - lat.mean(0))
    counts = rng.poisson(DEPTH * E / E.mean(0)).astype(float)
    return counts, grp


def partition_blocks(lr):
    """The leaf partition a landscape induces, as a set of blocks (label-name independent)."""
    lr = np.asarray(lr, dtype=object)
    return frozenset(frozenset(np.where(lr == u)[0].tolist()) for u in set(lr.tolist()))


def landscape(tree, branches, rng=None, shuffle=False):
    z = np.zeros(tree.n_nodes, bool)
    for b in branches:
        if b is not None:
            z[b] = True
    lr = leaf_regimes(tree, z)
    if shuffle:
        lr = np.array(rng.permutation(lr), dtype=object)
    return lr


def evidence(Z, tree, lr):
    """Per gene: the call, and dAICc of OUX against the better of BM1/OU1 (positive = favours OUX)."""
    node_regime, uniq = paint_regimes(tree, lr)
    fits = fit_models(Z, tree, node_regime=node_regime, n_regimes=len(uniq))
    if "OUX" not in fits:
        n = Z.shape[1]
        return np.array(["BM1"] * n, dtype=object), np.full(n, -np.inf)
    base = np.minimum(fits["BM1"]["aicc"], fits["OU1"]["aicc"])
    gain = base - fits["OUX"]["aicc"]
    names = ["BM1", "OU1", "OUX"]
    A = np.stack([fits[m]["aicc"] for m in names])
    call = np.array([names[i] for i in np.argmin(A, axis=0)], dtype=object)
    return call, gain


def main():
    rng = np.random.default_rng(7)
    tree = Tree.balanced(256, 1.0).scale_height(1.0)
    b1, b2 = two_clades(tree)
    counts, grp = simulate(tree, b1, b2, seed=0)
    X, t = scout_preprocess(counts, tree, normalize=True, smoothing_k=8)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)

    par, sib, kid = relatives(tree, b1)
    sets = tree.leaf_sets()
    size = len(sets[b1])
    distant = next(v for v in range(tree.n_nodes)
                   if tree.parent[v] >= 0 and abs(len(sets[v]) - size) <= size // 3
                   and jaccard(tree, v, b1) == 0 and jaccard(tree, v, b2) == 0)
    par2 = int(tree.parent[distant])
    true_blocks = partition_blocks(leaf_regimes(tree, _mask(tree, [b1, b2])))

    scenarios = [
        ("true (b1,b2)",         [b1, b2],       False),
        ("parent(b1) + b2",      [par, b2],      False),   # NB: same leaf partition as the truth
        ("parent(b1) only",      [par],          False),
        ("sibling(b1) + b2",     [sib, b2],      False),
        ("child(b1) + b2",       [kid, b2],      False),
        ("distant clade + b2",   [distant, b2],  False),
        ("both wrong",           [par2, distant], False),
        ("shuffled tip labels",  [b1, b2],       True),
    ]

    rows = []
    for label, branches, shuf in scenarios:
        lr = landscape(tree, branches, rng=rng, shuffle=shuf)
        call, gain = evidence(Z, tree, lr)
        supplied = branches[0]
        same_partition = partition_blocks(lr) == true_blocks
        row = dict(scenario=label, same_partition_as_truth=bool(same_partition),
                   overlap_with_true=round(jaccard(tree, supplied, b1), 3) if supplied is not None else np.nan,
                   n_regimes=len(set(lr)))
        for g in ("NULL-BM1", "NULL-OU1", "SIGNAL"):
            m = grp == g
            row[f"pct_OUX_{g}"] = float(np.mean(call[m] == "OUX"))
            row[f"median_gain_{g}"] = float(np.median(gain[m]))
        rows.append(row)
        print(f"{label:22s} {'SAME-PARTITION' if same_partition else '   different  '} "
              f"overlap={row['overlap_with_true']!s:>5}  "
              f"adaptive calls: NULL-BM1 {row['pct_OUX_NULL-BM1']:.0%}  "
              f"NULL-OU1 {row['pct_OUX_NULL-OU1']:.0%}  SIGNAL {row['pct_OUX_SIGNAL']:.0%}   "
              f"| median dAICc(OUX vs best) on SIGNAL = {row['median_gain_SIGNAL']:+.1f}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", "wrong_branches.csv")
    df.to_csv(out, index=False)
    print(f"\n[wrote {out}]")
    figdir = os.path.join(HERE, "results", "figures"); os.makedirs(figdir, exist_ok=True)
    figure(df, os.path.join(figdir, "wrong_branches.png"))
    tr = df[df.scenario == "true (b1,b2)"].iloc[0]
    print("\nRead-out:")
    print(f"  * spurious adaptive calls on genes with NO event, under a wrong-but-plausible landscape:")
    for _, r in df.iterrows():
        if r.scenario == "true (b1,b2)":
            continue
        print(f"      {r.scenario:22s} NULL-BM1 {r['pct_OUX_NULL-BM1']:.0%}  NULL-OU1 {r['pct_OUX_NULL-OU1']:.0%}")
    print(f"  * evidence for a WRONG branch vs the TRUE branch (median dAICc on SIGNAL genes):")
    for _, r in df.iterrows():
        print(f"      {r.scenario:22s} {r['median_gain_SIGNAL']:+8.1f}"
              + ("   <- truth" if r.scenario == "true (b1,b2)" else ""))


def figure(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    lab = [str(x).replace(" + ", "+\n") for x in df.scenario]
    xs = np.arange(len(df))
    same = df.same_partition_as_truth.values

    w = 0.27
    ax[0].bar(xs - w, df["pct_OUX_NULL-BM1"], w, color="#7f7f7f", label="NULL genes (BM1)")
    ax[0].bar(xs,     df["pct_OUX_NULL-OU1"], w, color="#bbbbbb", label="NULL genes (OU1)")
    ax[0].bar(xs + w, df["pct_OUX_SIGNAL"],   w, color="#d62728", label="SIGNAL genes")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(lab, rotation=35, ha="right", fontsize=7)
    ax[0].set_ylabel("fraction called adaptive (OUX)")
    ax[0].set_title("Adaptive calls under supplied landscapes\n(NULL genes have NO event: any call is spurious)")
    ax[0].legend(fontsize=8)

    cols = ["#2c7fb8" if s else "#d62728" for s in same]
    ax[1].bar(xs, df.median_gain_SIGNAL, color=cols)
    ax[1].axhline(0, color="k", lw=1); ax[1].axhline(2, ls=":", color="grey", lw=1)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(lab, rotation=35, ha="right", fontsize=7)
    ax[1].set_ylabel("median dAICc (OUX vs best of BM1/OU1)")
    ax[1].set_title("Evidence for adaptation at the SUPPLIED location\n"
                    "blue = same leaf partition as the truth (indistinguishable)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
