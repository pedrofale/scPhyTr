# Spatial phylogenetic comparative methods (heritable vs niche)

Spatial lineage tracing (e.g. **PEtracer**, Koblan et al. *Science* 2025 — MERFISH expression +
Cassiopeia trees) measures, for the same cells, a **lineage tree**, a **spatial coordinate**, and a
**count** expression profile. This raises a question the descriptive pipeline cannot answer cleanly:
for each gene, is its expression variation **heritable** (tracks the lineage) or **niche-driven**
(tracks spatial position)? Under local tumour growth these are *confounded* — spatial proximity ≈
phylogenetic proximity — so Moran's-I heritability and Hotspot spatial modules cannot separate them.

scPhyTr replaces the descriptive scatter with one generative model over (ancestry × space × counts).

## The model

For each gene (fit independently, univariate latent), leaf log-expression is an additive field

```
z = u + s + e ,      Y ~ NB(S · exp(z))      (Poisson if no dispersion)
  u ~ Brownian motion on the tree            (variance σ²_phylo)   — heritable / clonal
  s ~ spatial GMRF on the leaf kNN graph      (variance σ²_space)   — niche / microenvironment
  e ~ N(0, σ²_resid I)                         (iid leaf residual)   — off by default
```

`u` lives on the tree nodes, `s`/`e` on the leaf spatial graph; they meet only at the leaves through
the count likelihood. We fit the three variance scales by the **joint Laplace marginal** (sparse,
SuperLU-factorised, warm-started) and report `frac_heritable = v_phylo / (v_phylo + v_space)` from the
posterior component variances. See `src/scphytr/inference/spatial_decomposition.py`.

## API

```python
ph.pp.setup_anndata(adata, tree)                 # adata.obsm['spatial'] must hold per-cell coords
ph.pp.spatial_neighbors(adata, n_neighbors=8)    # leaf kNN GMRF precision -> uns['spatial_graph']

ph.tl.decompose_variance(adata, genes=panel)     # -> var['v_phylo','v_space','frac_heritable']
ph.pl.variance_decomposition(adata)              # the heritable–niche plane (replaces the PEtracer scatter)

ph.tl.spatial_programs(adata, genes=panel)       # gene-gene niche AND clonal correlation, deconfounded:
                                                 #   uns['niche_corr']  (spatial component, lineage removed)
                                                 #   uns['clonal_corr'] (phylo component, niche removed)
ph.tl.covariate_rate_shifts(adata, obs="niche")  # state-dependent BM rates for a discrete clone/niche label
```

Simulate ground truth with `ph.simulate_spatial_panel(tree, sigma2_phylo, sigma2_space, ...,
growth="diffuse"|"clonal", intermixing=..., spatial_lengthscale=...)`.

## Why it beats the descriptive pipeline (deconfounding)

The advantage is **axis-specific**: it appears on the low-SNR, confounded **tree** axis, not the
high-SNR spatial axis.

- **Per-gene heritability** (`analysis/benchmark/spatial_deconfounding.py`): scPhyTr separates
  heritable from niche genes at AUROC **0.99** vs Moran's-I 0.69 vs Hotspot 0.79 (pooled, realistic
  smooth-gradient regime). Naive autocorrelation collapses as niche gradients smooth; scPhyTr holds.
- **Gene-gene clonal modules** (`analysis/benchmark/spatial_clonal_modules.py`): recovering
  shared-lineage modules deconfounded from niche — scPhyTr **0.98** vs Hotspot-tree 0.76.
- **Spatial (niche) modules** (`analysis/benchmark/spatial_modules.py`): scPhyTr **matches** Hotspot
  (~0.97–0.99). The niche axis is unconfounded (smooth, high autocorrelation), so there is nothing to
  deconfound — an honest tie, not a win.

Mechanism: BM-on-a-tree is high-frequency (tree Moran's-I ~0.2 even when heritable), so tree
autocorrelation is low-SNR and confoundable; smooth niche gradients are low-frequency (Moran's-I
~0.7), so spatial autocorrelation is already robust.

## Why a count model (the sparse-depth advantage)

Independently of the spatial split, modelling *counts* (not `log1p(Y/S)` as a Gaussian trait) matters
at single-cell / MERFISH depth, where sampling noise masquerades as biological variance:

- **Heritability** (`depth_heritability.py`): at ~3 counts/cell (real PEtracer regime) scPhyTr recovers
  λ≈0.49 (true 0.6) vs Pagel's-λ 0.37 / Gaussian-on-log 0.29 — the Gaussian methods lose ~half the
  signal to attenuation. Converges by ~25 counts.
- **Co-evolution** (`depth_coevolution.py`): scPhyTr recovers the true correlation; Felsenstein
  contrasts / naive attenuate and plateau *below* truth even at high depth.
- **Plasticity** (`depth_plasticity.py`): a no-pseudobulk read-out competitors cannot compute; a naive
  within-clone variance conflates count sampling with biology (depth-dependent inflation), scPhyTr does
  not.

The count edge is scoped to low depth — exactly where MERFISH and this data live.

## Real data (PEtracer)

`analysis/petracer/` loads the deposited tumours (figshare 28473866) and runs the decomposition:
`load.py` (networkx Cassiopeia tree → ete3, raw counts + coords + covariates), `decompose_tumors.py`
(per-tumour decomposition, cached; validation; figure). On M2 tumours the split is bimodal and
reproducible across tumours: `Arg1` (hypoxic myeloid), `Sdc1`/`Nes`/`Vcan` (stroma/vasculature) come
out **niche**; immune/lineage-identity genes come out **heritable** — consistent with PEtracer's own
finding that tumour cell state is substantially lineage-heritable. The niche calls track independent
spatial evidence (per-gene `frac_heritable` vs spatial Moran's-I r = −0.68 on the 736-cell tumour; vs
distance-to-tumour-margin r = −0.46).

## Scalability & honest caveats

- **Gene axis (win):** the low-rank Poisson factor model (`K = WWᵀ`, k factors) fits 800 genes in ~3 s
  vs a full p×p covariance's hours (`runtime_scaling.py`).
- **Cell axis:** the tree-Laplace is O(cells) and, after the sparse/L-BFGS engine work, competitive
  with dense ML PCM at real tree sizes (≤~5k cells). It is **not** faster than BLAS-optimised dense
  Cholesky at small n — the honest runtime wins are the gene axis and vs MCMC (~40× on rate shifts).
- **Univariate v1:** genes are decomposed independently; a joint tree⊕space factor model is future work.
- **Identifiability:** the heritable/niche split degrades at very low clonal intermixing (lineage ≈
  space); reported, not hidden.
- **BM only:** `decompose_variance` uses Brownian motion for the heritable component (no OU).
