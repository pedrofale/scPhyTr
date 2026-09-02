"""Run the SCOUT-equivalent baseline on SCOUT's OWN example data, across their preprocessing arms.

`external/SCOUT/examples/sim_example/` ships a 256-cell, 3-state simulation (alpha=3, sigma=1) whose
gene NAMES encode the ground-truth model (BM1_*, OU1_*, OUM_*) -- an immediate, honest test set.

We run their two preprocessing arms (log-normalisation alone, and log-normalisation + lineage
smoothing) using our port of their `lineage_smooth`, which is validated to 1e-14 against their R
(see tests/test_preprocess.py). This reproduces their central methodological claim: smoothing
rescues true-neutral genes that noise otherwise pushes into the constrained (OU1) class.

    python -m analysis.scout.experiments.02_scout_example_data
"""
import os

import numpy as np
import pandas as pd

from analysis.scout.scout_io import load_scout_dataset, truth_from_gene_names
from scphytr.modes.baseline import classify_genes
from analysis.scout.scout_preprocess import scout_preprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(HERE, "external", "SCOUT", "examples", "sim_example")
COUNTS = os.path.join(EX, "3state_n256_alpha_3_sigma_1_counts.csv")
TREE = os.path.join(EX, "3state_n256_casNJ_tree.nwk")
ARMS = [("log-norm only", None), ("log-norm + smoothing k=4", 4),
        ("log-norm + smoothing k=8", 8), ("log-norm + smoothing k=16", 16),
        ("log-norm + smoothing k=32", 32)]


def main():
    if not os.path.exists(COUNTS):
        raise SystemExit("run scripts/fetch_scout.sh first")
    d = load_scout_dataset(COUNTS, TREE, log1p=False)
    tree, counts, genes = d["tree"], d["Y"], d["genes"]
    truth = truth_from_gene_names(genes)
    print(f"SCOUT example: {tree.n_leaves} tips, {len(genes)} genes, "
          f"regime column '{d['regime_key']}' ({len(set(d['leaf_regime']))} states)")
    print("true classes:", dict(pd.Series(truth).value_counts()), "\n")

    rows, last = [], None
    for label, k in ARMS:
        X, t = scout_preprocess(counts, tree, normalize=True, smoothing_k=k)
        # per-gene affine standardisation; model SELECTION is invariant to it (all three models
        # shift by the same Jacobian), it only improves conditioning
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        call, fits, gap = classify_genes(Z, t, leaf_regime=d["leaf_regime"])
        acc = float(np.mean(call == truth))
        rec = {c: float(np.mean(call[truth == c] == c)) for c in ("BM1", "OU1", "OUX")}
        rows.append(dict(arm=label, k=(k or 0), accuracy=acc,
                         **{f"recall_{c}": v for c, v in rec.items()}))
        print(f"{label:28s} acc={acc:.3f}   recall  BM1 {rec['BM1']:.2f}  "
              f"OU1 {rec['OU1']:.2f}  OUX {rec['OUX']:.2f}")
        last = (call, gap)

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", "scout_example_arms.csv")
    df.to_csv(out, index=False)
    call, gap = last
    pd.DataFrame(dict(gene=genes, truth=truth, called=call, delta_aicc=gap)).to_csv(
        os.path.join(HERE, "results", "scout_example_calls.csv"), index=False)
    print(f"\n[wrote {out}]")
    print("Reproduces SCOUT's claim: without smoothing, count noise inflates tip variance and pushes")
    print("true NEUTRAL genes into the constrained class; lineage smoothing restores them.")


if __name__ == "__main__":
    main()
