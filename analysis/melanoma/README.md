# B2905 melanoma 24-subline analysis (Hirsch et al. 2025, Cell Systems)

Running scPhyTr on the data from *Stochastic modeling of single-cell gene
expression adaptation reveals non-genomic contribution to evolution of tumor
subclones* (Hirsch et al., Cell Systems 16:101156, 2025), which used
[EvoGeneX](../../EvoGeneX) to detect adaptive gene expression across the clonal
sublines of a B2905 melanoma model. See `docs/01_methods.md` §10 for the method
comparison.

## Data

- **Expression**: the trisicell `sublines_scrnaseq` MuData release asset
  (175 QC-passed cells x 55,401 genes, full-length Smart-seq2; `tpm`/`fpkm`
  layers; per-cell `clone` = subline label). Downloaded to
  `data/external/sublines_scrnaseq.h5md.gz` (a plain HDF5/.h5mu file despite the
  name) from:
  ```bash
  curl -L -o data/external/sublines_scrnaseq.h5md.gz \
    https://github.com/faridrashidi/trisicell/releases/download/v0.0.0/sublines_scrnaseq.h5md.gz
  ```
- **Raw counts** (for the count model): scPhyTr models *counts*, but the
  trisicell asset only ships TPM/FPKM. The raw RSEM expected counts for the same
  Smart-seq2 cells are GEO **GSE215960** (the `[scRNA-seq]` subseries of the
  SuperSeries GSE215963, per the paper's data-availability statement):
  ```bash
  curl -L -o data/external/GSE215960_counts.tsv.gz \
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE215nnn/GSE215960/suppl/GSE215960_Expression_CountValue.tsv.gz
  ```
  192 cells x 55,401 genes; values are RSEM *expected* counts (fractional), which
  `load_counts` rounds to integers for the Poisson model. Cells are keyed by the
  same plate-well ids as the asset, so the asset's `cells->clone` map attaches
  subline labels (169 cells map to the 23 tree leaves; C2 dropped).
- **Phylogeny**: the consensus subline tree vendored with the paper's
  reproducibility repo,
  `nongenomic-evolution-of-tumor-subclones/tree_files/sc-bwes-cons-resolved-10.tree`
  (23 leaves; subline C2 is absent / excluded for technical reasons).
- **Regimes** (for the OU-2 adaptive test): the paper's chosen/background
  partitions in `nongenomic-evolution-of-tumor-subclones/regime_files/`
  (`har.csv`, `mas.csv`, `sas.csv`).

`load.py` reads the MuData with `h5py` (no `mudata` dependency needed),
aggregates cells to a per-subline pseudobulk (mean of `log2(1+TPM)`), aligns to
the tree leaves, and filters to genes expressed in all sublines (8,368 genes).

## Count model: cells within a subline

scPhyTr models counts, so the subline tree's leaves are *subclones* and the
~5-8 cells per subline are multiple observations on the same leaf latent
(`MultiCellPoissonObservation` in `inference/laplace.py`):

- **Pure Poisson** (`dispersion=None`): cells are exact replicates of the
  subline latent; the sufficient statistics collapse to per-subline *summed*
  counts (`load_subclone_counts`) -- equivalent to `PoissonObservation` on
  summed counts (validated).
- **Within-subline overdispersion** (`dispersion=r`): a Gamma-Poisson per-cell
  random effect models genuine intra-clonal heterogeneity (*plasticity*) beyond
  shot noise -- the count-level analogue of EVE's within-population variance.
  This also regularizes the tree's zero-length branches, so the epsilon-floor
  hack the pseudobulk smoke test used is unnecessary here.

`validate_multicell.py` checks both branches (pure Poisson == summed-count
Poisson; NB grad/curvature vs finite differences) and fits a multivariate BM
diffusion `K` from the real subline counts end-to-end (169 cells).

## Gene length (Smart-seq2)

Smart-seq2 is full-length, so read counts scale with transcript length. But gene
length is a per-gene constant and BM/OU give every gene a free baseline (root
mean / optimum) that absorbs it, so the evolutionary `K` (rates and gene-gene
correlations) is **invariant** to length -- and one must *not* divide the counts
by length (it would break the Poisson likelihood). The principled option, if you
want length explicit, is a tximport-style **length offset** `S_i * L_g`:
`load_counts(length_offset=True)`, with `L_g` derived from counts and FPKM
(`effective_lengths`). `length_invariance.py` confirms empirically that fitting
`K` with vs without the offset agrees to ~1e-11 (optimizer tolerance). What
length really affects is *precision*: low-count (incl. short) genes carry less
information, handled by the expression filter, not by length correction.

## Heritable vs plastic decomposition (`heritable_vs_plastic.py`)

`fit_mv_em(..., fit_dispersion=True)` now jointly fits the heritable diffusion
`K` (latent M-step) **and** the per-gene within-subclone NB dispersion `r`
(observation M-step, `MultiCellPoissonObservation.update_dispersion`). Per gene
this gives a heritable tip variance `K_gg * T` and a plastic within-subclone
variance `trigamma(r_g)`; their ratio is the count-level EVE variance ratio /
PATH heritability-plasticity axis. On the real data the most *plastic* genes are
ECM/invasion (Thbs1, Adamts1, Fn1) and the most *heritable* are housekeeping
(Gapdh, ...) -- consistent with the paper's plastic invasion program.

## OU-2 adaptive test (`adaptive_ou2.py`)

`load_regimes(tree, "har"|"sas"|"mas")` parses the paper's chosen/background
painting (internal nodes via MRCA of a descendant-leaf pair). `adaptive_ou2.py`
fits BM / OU-1 / OU-2 per gene to the multi-cell Poisson observation and selects
by AIC -- an OU-2 win = adaptively shifted expression in the chosen sublines, the
scPhyTr count-model analogue of the EvoGeneX adaptive test.

> **Fixed a silent-failure footgun.** `fit_mv_latent` caught the latent model's
> "zero-length branch" `ValueError` and returned `1e18` for every step, so on the
> un-floored tree it silently returned its *initialization* instead of a fit.
> `load_tree()` now floors zero-length branches everywhere, and `fit_mv_latent`
> raises if no optimizer step ever found a finite marginal.

## Adaptive scan results and the paper's Wnt claim

`adaptive_ou2.py` now folds the within-subclone dispersion into the test: a
per-gene NB `r` is estimated once (`update_dispersion` at the empirical
per-subclone means) and held fixed across BM/OU-1/OU-2, so the AIC comparison
isolates the between-subclone optimum shift. `adaptive_enrichment.py` then tests
the adaptive (OU-2) set for the paper's KEGG Wnt-signaling enrichment
(`kegg_wnt_mmu04310.txt`, fetched from KEGG REST) via Fisher's exact test;
`--targeted` scans the data's Wnt genes plus a size-matched random control for a
properly-powered version of the specific claim.

**HA-R, 150 HVGs:** 21 adaptive (OU-2), visibly dominated by proliferation /
DNA-replication genes (Cdk1, Mcm5, Rrm1, Rrm2, Prim1, Pclaf, Rad51ap1, Smc2,
Kif20b, Tcf19) -- a coherent program, but at 150 genes no single KEGG term is
significant and only 2 HVGs are Wnt (underpowered for the Wnt claim).

**HA-R, targeted Wnt-vs-control:** no Wnt enrichment (6/80 Wnt vs 2/14 control
adaptive; Fisher p=0.91). Caveat: the "detected in all sublines" filter dropped
106/120 control genes but only 95/175 Wnt, breaking the size-match -- so this is
not a clean test. The real signal is that adaptive rate tracks *expression
variability* (HVGs ~14% adaptive; broadly-expressed Wnt regulators ~7%), so Wnt
pathway membership per se doesn't predict adaptive expression at this scale. The
adaptive Wnt genes that do appear (Dvl2, Nlk, Ppp3ca) skew *non-canonical* -- the
paper's direction for the resistant regime -- but n=6 is far too small to claim it.

**Honest status:** the count-native, dispersion-aware OU-2 machinery is
reproduced and gives coherent biology, but a faithful enrichment match to the
paper is not achievable at the scan scale feasible here (per-gene Nelder-Mead OU
is ~4-10 s).

## Making the per-gene scan fast (toward transcriptome-wide)

Diagnosis (profiling): for `p=1` the *multivariate* path wasted almost all its
time in scipy `cho_factor`/`cho_solve` on **1x1 matrices** (~72k LAPACK calls per
OU fit, pure validation overhead). Two levels of fix:

1. **Scalar univariate path (done, ~2x).** The scan now fits via
   `fit_bm_counts` / `fit_ou_counts` / `fit_ou_regimes_counts`
   (`tree_laplace.py`, plain scalar arithmetic, no LAPACK) with a 1-D-aware
   `MultiCellPoissonObservation(..., univariate=True)`. ~2.7 -> ~1.4 s/gene. Also
   hardened: those count-fits now catch the degenerate `sigma2 -> 0` case (the
   latent model raises on `v*sigma2 <= 1e-12`) instead of crashing the scan.

2. **Batch over the gene axis (prototype, the real win).** The remaining cost is
   Python loops over tree nodes, repeated per optimizer-eval per gene -- but those
   loops are identical across genes. Carrying a gene axis makes each per-node
   scalar a vectorized `(G,)` op, so the loop is paid *once* for all genes.
   `batched_prototype.py` implements the batched BM Laplace marginal: it matches
   the per-gene path to ~1e-8 and computes **300 genes in 1.5 ms (0.005 ms/gene)**
   vs ~1.4 s/gene -- and G=1 costs the same as G=300. This is what makes a
   transcriptome-wide *per-gene* model-selection scan feasible; it complements
   **phylogenetic factor analysis** (`poisson_factor.py`, shared k factors,
   O(Nk^3)) which is the scalable route for the latent representation itself.

## Batched engine (`scphytr.inference.batched`, `batched_scan.py`)

`BatchedTreeLaplace` fits BM / OU-1 / OU-2 for **all genes at once**: batched
Newton mode-find + Laplace marginal vectorized over the gene axis, optima `theta`
profiled in closed form at the mode, `(alpha, sigma2)` grid-maximized. Validated:
the batched marginal matches the per-gene scalar path to **~1e-8**. Speed: **~6
ms/gene** (G=5000 in 30 s; a transcriptome ~2 min) -- ~200x over the per-gene
path, and per-gene cost keeps dropping with G. This is the scalable per-gene
model-selection engine; phylogenetic factor analysis (`poisson_factor.py`) is the
complementary route for the shared latent representation.

**Calibration caveat (open):** the batched engine uses pure-Poisson *summed*
counts, so within-subclone overdispersion is handled by a crude per-gene
**quasi-Poisson** dispersion (Pearson phi). For high-count genes phi blows up
(~hundreds), so adaptive calls are not yet well-calibrated (~19% OU-2, likely
over-called). The principled fix is the per-cell **NB** leaf reduction in the
batched engine (carry cells, not just summed counts) -- the count analogue of the
multi-cell NB obs already used in the per-gene scan.

## Open / next

- NB per-cell leaf reduction in the batched engine for calibrated adaptive calls.
- A *detection-matched* random control so the Wnt test is fair; the regime
  contrast (HA-R resistant -> non-canonical vs SA-S sensitive -> canonical).
