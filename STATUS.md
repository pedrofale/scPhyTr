# scPhyTr manifest
Felsenstein: https://www.jstor.org/stable/2461605?seq=15
Going from descriptions of trees to evolutionary model-based understanding with phylogenetic comparative methods.

Tumor phylogeography: https://www.nature.com/articles/s41467-018-04724-5, https://www.nature.com/articles/s41592-024-02438-9
Boring!

TreeVAE: https://www.biorxiv.org/content/10.1101/2021.05.28.446021v1.full.pdf
My project is an extension of this to learn evolutionary parameters of the trait model. Latent space.

CAGEE: https://academic.oup.com/mbe/article/40/5/msad106/7157541#406265594
A trait model for gene expression. Allows explicit groups of genes to be grouped and have the same parameter. Perhaps a hierarchical model would be better, or a sparse GP type of approach?

Incorporating selection with the OU model

Estimate state-dependent rates, using spatial data to infer states.

## Roadmap: realistic factor-analysis baselines (KP-Tracer, docs/04_real_data_kptracer.md)
The current real-data study uses the *fast/simple* setup (log-normalized expression,
linear-Gaussian factor analysis, one-switch C-vs-I comparison). Planned upgrades to
make it realistic, keeping in mind:
- **Poisson observation model.** Replace the log-normal approximation with a Poisson
  (log-normal) factor model on raw counts, marginalized via the latent tree-Laplace/EM
  machinery (docs/01_methods.md, docs/02_inference_engines.md). Latent evolutionary model
  stays independent of the observation model.
- **NMF as a baseline.** Add a nonnegative matrix factorization baseline (scDEF-style)
  for the naive side, and build a phylo-aware NMF (nonnegative loadings + a tree prior
  on factor scores). Note: the "FA on Felsenstein contrasts" trick does NOT carry over,
  since whitening breaks nonnegativity — needs a genuine tree-prior NMF fit.
- Per-tumor -> hierarchical model sharing programs across tumors with per-tumor trees.

## Hotspot gene modules are tree-confounded (docs/04_real_data_kptracer.md §4)
Demonstrated that Hotspot (DeTomaso & Yosef 2021) gene modules inherit the
Felsenstein confounding: its local-correlation Z is scored against a
cell-exchangeability null the phylogeny violates, so it estimates a smoothed tip
covariance (~ C ⊗ K) rather than the evolutionary covariance K.
- Ground-truth control (hotspot_confounding_sim.py): on independent genes on a real
  tree, Hotspot calls modules 100% of the time; contrast-based K estimate ~5%.
- Real data (hotspot_vs_phylo_real.py): across 10 tumors, module shrinkage on
  deconfounding tracks clade eta^2 (Spearman 0.84); clonal modules lose ~35% of their
  coherence (worst case 0.47->0.16), cell-state modules ~4%.
- Natural next step: report the deconfounded K module structure directly (block
  structure of K via PFA) as the scPhyTr alternative to Hotspot modules; and a
  phylo-aware NMF for nonnegative modules (ties into the NMF roadmap item above).

