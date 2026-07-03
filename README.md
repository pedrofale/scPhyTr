# scPhyTr
Phylogenetic comparative methods for single-cell gene expression data. `scPhyTr` implements Brownian Motion, Ornstein-Uhlenbeck and peak-shift model fitting algorithms for single-cell phylogenies and gene expression data to infer evolutionary rates of genes or pathways, evolutionary correlations, and clade-specific rate variation. scPhyTr may also take known covariates to infer covariate-driven rate changes. Additionally, if spatial data is available, scPhyTr can identify niches with different evolutionary rates for any trait de novo.

Testing hypothesis about spatial effects of trait evolution?

Comparing to RevBayes, scPhyTr is tailored for the specificities of single-cell data, and extends its functionality to include variation in evolutionary rates in unknown clades, as well as the ability to extract covariates from spatial data to associate with rate variations.

To install scPhyTr and get started, read the documentation.

## Usage

`scPhyTr` has a scanpy-style API (`pp` / `tl` / `pl`) over a modular backend
(*trait model* × *observation model* × *inference algorithm*). The leaves of the tree are
cells (or subclones); cells are kept as **subclonal replicates** of their leaf and are never
collapsed to pseudobulk.

```python
import scphytr as ph
import anndata

adata = anndata.read_h5ad("...")   # cells × genes; adata.obs['species'] = each cell's leaf label
tree  = ...                        # a scphytr Tree over the leaves

# --- preprocessing ---
ph.pp.setup_anndata(adata, tree)              # attach tree + per-cell leaf index + size factors
adata = ph.pp.cut_tree(adata, min_cells=5)    # prune sparse leaves

# --- tools: read-outs (default inference = fast Laplace; method= will select MCMC/VI later) ---
ph.tl.estimate_rate(adata, genes=panel)                  # per-gene BM rate σ²  -> adata.var['rate']
ph.tl.heritability(adata, genes=panel)                   # Pagel's λ            -> adata.var['lambda','lambda_p']
ph.tl.plasticity(adata, genes=panel)                     # heritable vs within-clone plastic variance (needs replicates)
ph.tl.detect_adaptive(adata, genes=panel)                # BM vs OU vs OU2 (adaptive) per gene
ph.tl.detect_rate_shifts(adata, character="Mki67")       # de-novo clade rate shifts -> adata.uns['rate_shifts']
ph.tl.evolutionary_correlation(adata, genes=panel)       # deconfounded K       -> adata.uns['K','K_corr']
ph.tl.factor_analysis(adata, k=5, genes=panel)           # phylogenetic factor analysis -> adata.uns['pfa']

# --- spatial read-outs (when adata.obsm['spatial'] and a lineage tree are both present) ---
ph.pp.spatial_neighbors(adata, n_neighbors=8)            # leaf kNN spatial graph (GMRF) -> adata.uns['spatial_graph']
ph.tl.decompose_variance(adata, genes=panel)             # split each gene's variance into HERITABLE (tree) vs
                                                         #   NICHE (space) -> adata.var['v_phylo','v_space','frac_heritable']
ph.tl.spatial_programs(adata, genes=panel)               # gene-gene niche vs clonal correlation -> uns['niche_corr','clonal_corr']
ph.tl.covariate_rate_shifts(adata, obs="niche")          # state-dependent rates for a discrete covariate (clone/niche)

# --- plotting ---
ph.pl.rate_tree(adata)                        # the tree, clades coloured by their fitted rate (shifts starred)
ph.pl.plot_tree(adata, color="Mki67")         # the tree, leaves coloured by a gene / obs column
ph.pl.matrix(adata, "K_corr")                 # heatmap of the evolutionary correlation
ph.pl.loadings(adata)                         # phylogenetic-factor-analysis gene loadings
ph.pl.variance_decomposition(adata)           # heritable–niche plane (the deconfounded replacement for the PEtracer scatter)

# usual scanpy plots work on anything scPhyTr writes to .var / .obs
import scanpy as sc
sc.pl.umap(adata, color="rate")
```

Don't have data yet? Simulate it: `ph.simulate_panel(tree, K, ...)` (correlated multi-gene counts) or
`ph.simulate_spatial_panel(tree, sigma2_phylo, sigma2_space, ...)` (tree⊕space latent field + NB counts,
diffuse or Cassiopeia clonal-territory growth). See `notebooks/scphytr_tutorial.ipynb` (incl. the spatial
section) and `notebooks/scphytr_roundtrip.ipynb`.

**Status.** Implemented and validated: the above `pp` / `tl` / `pl` calls and the simulators, backed by the
linear-time exact-marginal `Laplace` inference (sparse, warm-started; scales to 10³–10⁴-cell trees). The
spatial `decompose_variance` has been applied to real spatial lineage-tracing data (PEtracer); see
`docs/05_spatial.md`. Planned (modular contract is in place): selectable Bayesian inference
(`method="mcmc"|"vi"|"pic"`).
