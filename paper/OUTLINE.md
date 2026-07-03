# scPhyTr — paper outline & framing (target: *Genome Biology*)

**Status:** locked framing, 2026-07-03. Supersedes the framing of `paper/main.tex` (the old
"exact linear-time inference" ML-style draft), which is the WRONG framing for GB and must be
inverted (engine → Methods, spatial decomposition → headline).

## One-liner
scPhyTr decomposes single-cell expression in **spatial lineage-traced** tissue into a **heritable
(clonal)** and a **niche (microenvironmental)** component, *deconfounding lineage from location* —
which the descriptive autocorrelation tools used in the field (Moran's I, Hotspot) cannot do,
because under local tumour growth lineage proximity ≈ spatial proximity.

## Why Genome Biology (framing rules)
GB rewards: a real biological question, a tool that **beats/complements existing tools in
head-to-head benchmarks**, a **real-data application with biological payoff**, and released,
usable software. It does NOT reward "linear-time exact inference" as a headline. So: biology-forward
throughout; the count-native linear-time engine is *Methods* (what makes it scale to real MERFISH
trees), never the pitch.

## Title candidates
1. Disentangling heritable and microenvironmental gene-expression programs in spatial
   lineage-traced tumours
2. scPhyTr: deconfounding clonal inheritance from niche in spatial single-cell lineage tracing
3. Heritable versus niche: a generative decomposition of expression in spatially resolved cell
   phylogenies

## Abstract skeleton (~150 words)
- Setup: spatial lineage tracing (PEtracer, MERFISH + Cassiopeia trees) measures lineage, location
  and expression in the same cells.
- Problem: is a program's spatial coherence because it is **clonally inherited** or because it
  **responds to the niche**? Under local growth the two are confounded; Moran's-I heritability and
  Hotspot modules cannot separate them.
- Method: scPhyTr, a generative latent-Gaussian model jointly over (tree × space × counts) that
  partitions each gene's variance into heritable + niche + residual, fit by a count-native
  linear-time Laplace engine that scales to thousands of cells.
- Results: on simulations it recovers the split and is calibrated; it deconfounds where Moran's-I /
  Hotspot are confounded (AUROC 0.99 vs 0.69/0.79); on real PEtracer tumours it separates heritable
  cell-identity programs from niche programs (hypoxia/vascular/stroma), replicated across tumours,
  and [HEADLINE BIOLOGICAL FINDING — to lock on full data].
- Software: scanpy-like, open source.

## Contributions (state these explicitly)
1. **Conceptual:** a generative model that jointly explains lineage + space + counts and
   deconfounds heritable from niche variation per gene — the first principled replacement for the
   descriptive heritable-vs-spatial scatter.
2. **Benchmarked deconfounding:** demonstrated superiority over the field's tools (Moran's-I,
   Hotspot) on the tree/clonal axis, with a mechanistic explanation (low-SNR tree axis).
3. **Real-data application:** first PCM decomposition of real spatial lineage tracing (PEtracer),
   recovering + extending its conclusions with a validated biological readout.
4. **Software/engine (enabler):** count-native, no-pseudobulk, linear-time sparse inference that
   makes the above tractable on real 10³–10⁴-cell trees; a unified toolkit (heritability, rate
   shifts, co-evolution) around the same fit.

## Narrative arc
confound (local growth ⇒ lineage≈space, so autocorrelation is ambiguous) → generative
tree⊕space⊕count model that separates them → simulations show recovery + calibration → head-to-head
deconfounding wins vs Moran's-I/Hotspot (+ mechanism) → real PEtracer: reproducible heritable/niche
split validated against independent spatial evidence → biological payoff → the method is a released,
scalable tool.

## Figure plan (GB = figure-driven; ~5 main + supplement)
- **Fig 1 — Concept & model.** (a) the confound cartoon (local growth ⇒ lineage≈space);
  (b) the generative model schematic z = u(BM tree) + s(spatial GMRF) + e, NB count decoder;
  (c) the decomposition output (a gene → heritable/niche/residual). *Status: BUILD (schematic).*
- **Fig 2 — Why a count model: it resists the attenuation that cripples Gaussian/noiseless-trait
  PCM at single-cell depth.** (a) heritability vs sequencing depth: scPhyTr flat near truth, Pagel's
  λ / Gaussian-on-log attenuate to <½ the true value at MERFISH depth (`depth_heritability.py`);
  (b) gene-gene **co-evolution** vs depth: scPhyTr recovers true ρ, Felsenstein contrasts / naive
  attenuate and plateau below truth even at high depth (`depth_coevolution.py`); (c) calibration /
  recovery vs planted truth (from spatial_decomposition.py). This is the demonstrated NON-SPATIAL
  advantage — the count likelihood, on the same sparsity axis as the spatial story. *Status: DONE —
  3 robust committed benchmarks: heritability (b250607), co-evolution (1b10a12), plasticity recovery
  (b5d5f49, fair version). Assemble. (A 4th, weak-selection detection, was built then DROPPED — not
  reproducible run-to-run; selection detection has genuinely little count-model advantage.)*
- **Fig S/2d — Scalability (honest).** Gene axis WIN: low-rank Poisson factor model (K=WWᵀ) fits 800
  genes in ~3s vs the full pxp covariance's ~32h-extrapolated (415x at 50 genes; `runtime_scaling.py`,
  commit 1fd47a7). Plus vs-MCMC (RevBayes ~40x). HONEST CAVEAT (do NOT claim): scPhyTr is NOT faster
  than dense ML PCM on the CELL axis until ~10^4 cells (large optimizer constant); measured 42s vs
  2.6s at 3500 cells. So frame runtime as gene-axis + vs-MCMC only.
- **Fig 3 — Deconfounding beats the field's tools.** (a) per-gene heritability: scPhyTr 0.99 vs
  Moran's-I 0.69 vs Hotspot 0.79 + smoothness sweep; (b) gene-gene clonal modules: scPhyTr 0.98 vs
  Hotspot-tree 0.76 (benchmark 3); (c) mechanism panel (tree axis low-SNR: BM Moran's-I ~0.2 vs
  niche ~0.7). *Status: HAVE (benchmarks 1 & 3, sweep csvs); assemble.*
- **Fig 4 — Real PEtracer: reproducible, validated split.** per-tumour frac_heritable distributions;
  validation scatter (frac vs spatial Moran's I, r<0; vs tumour_boundary_dist); cross-tumour
  reproducibility; heritable vs niche gene sets. *Status: PROTOTYPE (decompose_tumors.py, 2 M2
  tumours); NEEDS full data (all tumours, big-tree solver) + NB dispersion.*
- **Fig 5 — Biological payoff.** the deconfounded niche programs (spatial_programs / clonal
  modules) mapped to biology; a call the descriptive pipeline gets wrong that scPhyTr corrects;
  ideally a spatial-covariate rate shift or a program whose heritability depends on niche.
  *Status: BUILD on full data — this is the headline biology, still open.*
- **Supplement:** the broader toolkit as breadth (rate shifts vs RevBayes ~100× faster;
  deconfounded co-evolution; OU/selection), runtime/scaling, robustness.

## Section structure (GB: results-first)
Background → Results (Figs 1–5 as subsections) → Discussion → Conclusions → Methods (model;
count-native linear-time sparse Laplace engine; decomposition estimator; simulation & benchmarking
protocols; PEtracer data processing) → Availability of data and materials.

## Comparators (benchmarks central for GB)
Moran's I (PEtracer's heritability statistic), Hotspot (spatial & tree modes; their module tool),
naive correlation, RevBayes (rate shifts, supplement). Show scPhyTr (a) recovers their qualitative
conclusions and (b) corrects specific confounded calls.

## Open decisions
- **D1 — headline biology (Fig 5).** What is THE finding? Candidate: heritable = cell-identity/
  differentiation programs; niche = hypoxia (Arg1)/vascular (Clec14a,Sox17)/stroma (Vcan,Sdc1);
  and specific programs Hotspot mis-labels as spatial that are actually clonal. Needs full-data
  analysis to lock.
- **D2 — scope of the non-spatial toolkit.** Keep heritability/rate-shifts/co-evolution as a
  breadth section or push mostly to supplement? Lean: supplement + one breadth paragraph, so the
  spatial story stays the spine.
- **D3 — one paper vs companion methods paper.** Locked: ONE GB paper; do not split (engine alone
  is out-of-scope framing for GB).

## Work-to-submission checklist (each item feeds a figure)
1. **Big-tree solver** (symbolic-factorization reuse / better outer optimizer) → unlock full M2
   (incl. 5216-cell tree) + M1/M3 → Fig 4/5. *[paper-blocking]*
2. **NB dispersion** on MERFISH (we ran Poisson) + robustness check → Fig 4.
3. **Full-data decomposition** across all tumours; lock the biological finding → Fig 5 (D1).
4. **Assemble Figs 2 & 3** from existing benchmark outputs.
5. **Fig 1** schematic.
6. **Software polish:** installable package + tutorial notebook + data-availability.
7. **Reframed write-up** (invert main.tex).
