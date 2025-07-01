# scPhyTr
Phylogenetic comparative methods for single-cell gene expression data. `scPhyTr` implements Brownian Motion, Ornstein-Uhlenbeck and peak-shift model fitting algorithms for single-cell phylogenies and gene expression data to infer evolutionary rates of genes or pathways, evolutionary correlations, and clade-specific rate variation. scPhyTr may also take known covariates to infer covariate-driven rate changes. Additionally, if spatial data is available, scPhyTr can identify niches with different evolutionary rates for any trait de novo. 

Comparing to RevBayes, scPhyTr is tailored for the specificities of single-cell data, and extends its functionality to include variation in evolutionary rates in unknown clades, as well as the ability to extract covariates from spatial data to associate with rate variations.

To install scPhyTr and get started, read the documentation.

## Usage
import scphytr as ph
import anndata

adata = anndata.read_h5ad()

ph.tl.estimate_global_rate(adata, character=, model=) # Populates the adata.uns with a global rate for the specified character using the specified model
ph.tl.estimate_lineage_rates(adata, lineage=) # Populates the adata.uns with a rate for each lineage
ph.tl.estimate_state_rates(adata, state=) # Populates the adata.uns with a rate for each state
ph.tl.estimate_evolutionary_correlation(adata, characters=) # Populates the adata.uns with a correlation matrix for the specified characters
ph.tl.estimate_evolutionary_optimum(adata, character=) # Populates the adata.uns with an evolutionary optimal value for the specified character using an OU model
ph.tl.estimate_lineage_evolutionary_optima(adata, character=, lineage=) # Populates the adata.uns with an evolutionary optimal value for the specified character using an OU model in each lineage