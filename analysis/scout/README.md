# SCOUT comparison

The comparison machinery behind `scphytr.modes` / `tl.detect_modes` — see
`docs/06_mode_detection.md` for the method itself.

SCOUT (Stuart & McKenna, bioRxiv 2025.11.12.688020) fits BM1 / OU1 / OUx independently per gene on a
single-cell lineage tree, with the regime partition **supplied** by the user, and picks the
minimum-AICc winner. It is the closest published method and the thing to beat.

## What is here

| Path | What |
|---|---|
| `scout_preprocess.py` | port of SCOUT's preprocessing, validated to **1e-14** against their own R |
| `scout_io.py` | read SCOUT's `counts.csv` + newick, write our simulations in their format |
| `stan_scout.py`, `stan/scout.stan` | SCOUT's per-gene question restated as a Bayesian model comparison |
| `stan/bayesian_scout.pdf` | its typeset specification, including the BM recovery study |
| `stan/three_models.pdf` | the BM1 / OU1 / OUx equations on two pages |
| `scripts/` | fetch and run the **original** SCOUT R package, and its LUAD data |
| `experiments/` | 01–09, the numbered development experiments |
| `notes/SCOUT_digest.md` | what SCOUT does, precisely, and where the gap is |
| `results/` | the findings as CSV, plus the figures that were reasoned from |
| `sparseou-history.bundle` | the full development history — `git clone` it |

## Three preprocessing arms, and why it matters

On SCOUT's own example data (90 genes, truth encoded in the gene names), three-way accuracy:

| arm | accuracy |
|---|---|
| log-normalisation only | 0.589 |
| + lineage smoothing, k=8 (their default) | **0.889** |
| their EM variant (62 min) | **0.889** |

Smoothing is doing all the work, and EM buys nothing over it. The EM arm's residual errors are
one-sided: 8 of 30 constrained genes are called **adaptive**.

That reproduces their central methodological claim — and it is also the warning. Lineage smoothing
multiplies expression by a similarity matrix, so the smoothed covariance is `S Σ S'`, which no OU
can produce. Fed to a *generative* model it is absorbed as signal: 16 spurious events where truth
had one. `detect_modes` fits a noise term instead and never sees smoothed input.

## Reproducing

```bash
bash scripts/fetch_scout.sh          # clone SCOUT, ~360 MB of R deps into external/
bash scripts/setup_scout_env.sh      # the R environment (CRAN binaries; conda-forge lacks OUwie)
bash scripts/fetch_luad.sh           # their lung data, GEO GSE161363, 319 MB
```

`external/` and `data/` are re-fetchable and are not committed. `results/` holds the outputs that
are not: the CSVs are small and are the actual findings, while the large intermediates (`.rds`,
`.nc`, most figures) were dropped as regenerable.

## Provenance

Developed as the standalone `sparseou` repository, absorbed into scPhyTr on 2026-09-02 once it was
clear it was one method inside scPhyTr's remit rather than a separate project. The original
repository was deleted after `sparseou-history.bundle` was verified to restore an identical commit
set; that bundle is the only copy of its 7 commits, which were never pushed to a remote.
