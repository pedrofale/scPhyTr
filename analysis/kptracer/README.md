# KP-Tracer real-data analysis

End-to-end comparison of phylogeny-naive vs. phylogeny-aware factor analysis on
the Weissman-lab KP-Tracer mouse lung-adenocarcinoma lineage-tracing data. See
[`docs/04_real_data_kptracer.md`](../../docs/04_real_data_kptracer.md) for the writeup.

## Get the data

The processed data (~1.3 GB) is on Zenodo (record `5847462`):

```bash
mkdir -p data/external && cd data/external
curl -L -o KPTracer-Data.tar.gz \
  "https://zenodo.org/records/5847462/files/KPTracer-Data.tar.gz?download=1"
tar xzf KPTracer-Data.tar.gz KPTracer-Data/expression KPTracer-Data/trees
```

This yields `data/external/KPTracer-Data/expression/adata_processed.nt.h5ad`
(integrated sgNT AnnData) and `data/external/KPTracer-Data/trees/{tumor}_tree.nwk`
(per-tumor Newick). `data/external/` is git-ignored.

Requires `anndata` and `ete3` (already a scphytr dependency); `anndata` is enough
to read the `.h5ad` (no full scanpy needed). The Hotspot comparison additionally
needs `hotspotsc` (`pip install hotspotsc`).

## Run

```bash
python -m analysis.kptracer.inspect_data            # structural sanity check
python -m analysis.kptracer.semisynth_real_tree     # Felsenstein control on real tree
python -m analysis.kptracer.summary_figure          # naive vs phylo on real expression
python -m analysis.kptracer.hotspot_confounding_sim # Hotspot invents modules (ground truth)
python -m analysis.kptracer.hotspot_vs_phylo_real   # Hotspot modules vs deconfounded K (real)
```

Figures land in `analysis/kptracer/figures/`.

> **Numba cache note.** Hotspot uses numba; if running under a restricted
> filesystem set `NUMBA_CACHE_DIR` to a writable path (e.g.
> `export NUMBA_CACHE_DIR=$PWD/.numba_cache`).

## Files

| File | Role |
|---|---|
| `load.py` | load a tumor: tree + log-norm HVG expression, fast shared-time covariance `C` |
| `phylo_factor_utils.py` | contrasts, Horn parallel analysis, clade partition / `eta^2` |
| `inspect_data.py` | one-off structural inspection of the AnnData + trees |
| `semisynth_real_tree.py` | ground-truth Felsenstein control on the real topology |
| `real_data_compare.py` | naive vs phylo FA on one tumor's real expression |
| `summary_figure.py` | cross-tumor summary |
| `hotspot_utils.py` | Hotspot wrappers (tree/KNN) + naive/contrast gene-gene correlation |
| `hotspot_confounding_sim.py` | ground-truth control: Hotspot invents modules from the tree |
| `hotspot_vs_phylo_real.py` | real modules vs deconfounded correlation, across tumors |
