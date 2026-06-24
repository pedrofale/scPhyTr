"""Loader for the B2905 melanoma 24-subline data (Hirsch et al. 2025, Cell Systems).

The expression data is the trisicell ``sublines_scrnaseq`` MuData release asset
(``data/external/sublines_scrnaseq.h5md.gz``, actually a plain HDF5/.h5mu file);
the subline phylogeny is the consensus tree vendored with the paper's
reproducibility repo (``nongenomic-evolution-of-tumor-subclones/tree_files/``).

EvoGeneX treats the per-subline single cells as biological *replicates*. The
current scPhyTr Gaussian-trait path attaches one value per tree leaf, so the
natural first mapping is a per-subline pseudobulk: mean of log2(1+TPM) over the
cells of each subline. ``load_pseudobulk`` returns that table aligned to the
tree leaves; ``load_cells`` returns the full per-cell table (cells x genes) plus
the subline label, for replicate-aware or cell-as-leaf modeling later.
"""
from __future__ import annotations

import os
import h5py
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
H5MU = os.path.join(_ROOT, "data", "external", "sublines_scrnaseq.h5md.gz")
# Raw RSEM expected counts for the same Smart-seq2 cells (GEO GSE215960,
# the [scRNA-seq] subseries of the SuperSeries GSE215963). The trisicell
# MuData asset only ships TPM/FPKM; scPhyTr's Poisson model needs counts.
COUNTS = os.path.join(_ROOT, "data", "external", "GSE215960_counts.tsv.gz")
FPKM = os.path.join(_ROOT, "data", "external", "GSE215960_fpkm.tsv.gz")
TREE = os.path.join(
    _ROOT, "nongenomic-evolution-of-tumor-subclones",
    "tree_files", "sc-bwes-cons-resolved-10.tree",
)


def _decode(obs, key):
    """Decode an AnnData-in-HDF5 obs column, resolving categoricals."""
    codes = obs[key][:]
    cats = obs.get("__categories", {})
    if key in cats:
        levels = [c.decode() if isinstance(c, bytes) else c for c in cats[key][:]]
        return np.array([levels[i] if i >= 0 else "NA" for i in codes])
    return np.array([x.decode() if isinstance(x, bytes) else x for x in codes])


def load_cells(layer: str = "tpm"):
    """Return (X, genes, clone) for all 175 QC-passed cells.

    X : (n_cells, n_genes) float array of the requested layer (default TPM).
    genes : array of ``ensembl_symbol`` ids. clone : per-cell subline label.
    """
    with h5py.File(H5MU, "r") as f:
        exp = f["mod"]["expression"]
        X = exp["layers"][layer][:]
        genes = np.array([g.decode() if isinstance(g, bytes) else g
                          for g in exp["var"]["genes"][:]])
        clone = _decode(exp["obs"], "clone")
    return X, genes, clone


def tree_leaves():
    """Subline names at the leaves of the vendored consensus tree (C2 absent)."""
    from scphytr.utils.tree import Tree
    return list(Tree(TREE).phylotree.get_leaf_names())


def load_tree(floor_frac=0.01):
    """Load the subline tree with zero-length branches floored to a small epsilon.

    The consensus tree has 11 zero-length branches (polytomy-resolution artifacts,
    e.g. ``C21:0``). The latent-tree model requires positive branch lengths (a
    zero branch means parent==child with infinite precision), so we floor every
    branch -- including the root -- to ``floor_frac`` of the root-to-tip height.
    Returns a ``scphytr.utils.tree.Tree``. (NB: without this the optimizer's
    error-guard silently returns the initialization instead of a real fit.)
    """
    from scphytr.utils.tree import Tree
    tree = Tree(TREE)
    H = float(tree.root.get_farthest_leaf()[1])
    eps = floor_frac * H
    for nd in tree.phylotree.traverse():
        if nd.dist < eps:
            nd.dist = eps
    return tree


def load_pseudobulk(min_sublines: int | None = None, log: bool = True):
    """Per-subline pseudobulk table (sublines x genes), aligned to the tree.

    Aggregates cells to the mean of log2(1+TPM) per subline (EvoGeneX's
    per-gene log transform), keeps only sublines present in the tree, and (by
    default) genes expressed in every retained subline -- mirroring the paper's
    "non-zero in all species" filter.
    """
    X, genes, clone = load_cells("tpm")
    if log:
        X = np.log2(1.0 + X)
    leaves = tree_leaves()
    df = pd.DataFrame(X, columns=genes)
    df["__subline"] = clone
    pb = df.groupby("__subline").mean()
    pb = pb.reindex([s for s in leaves if s in pb.index])  # tree order
    if min_sublines is None:
        min_sublines = pb.shape[0]
    # drop genes not expressed (mean>0) in at least `min_sublines` sublines
    keep = (pb > 0).sum(axis=0) >= min_sublines
    return pb.loc[:, keep]


def _cell_to_clone():
    """Map each QC-passed cell id (plate-well) to its subline, from the asset."""
    with h5py.File(H5MU, "r") as f:
        obs = f["mod"]["expression"]["obs"]
        cells = _decode(obs, "cells")
        clone = _decode(obs, "clone")
    return dict(zip(cells, clone))


_REGIME_DIR = os.path.join(
    _ROOT, "nongenomic-evolution-of-tumor-subclones", "regime_files")


def load_regimes(tree, name):
    """Parse a paper regime file into a per-node painting for OU-2.

    The paper's ``regime_files/{name}.csv`` (``name`` in {har, sas, mas, single})
    label every tree node "1_chosen" or "2_background", identifying internal
    nodes by a pair of descendant leaves ``(node, node2)`` via their MRCA. We map
    each row to an ete3 node and return ``(regimes, n_regimes)`` with
    ``regimes[node] = 1`` for chosen, ``0`` for background -- the input to the
    multi-regime OU (``paint_regimes``-style) used by ``detect_adaptive_counts``
    and ``fit_mv_latent(regimes=...)``. The "chosen" sublines are the high-/low-
    adapting group the paper tests for an optimum shift.
    """
    df = pd.read_csv(os.path.join(_REGIME_DIR, f"{name}.csv"))
    pt = tree.phylotree
    regimes = {nd: 0 for nd in pt.traverse()}
    for _, row in df.iterrows():
        chosen = 1 if "chosen" in str(row["regime"]) else 0
        a = str(row["node"])
        b = row.get("node2")
        if pd.isna(b) or str(b) == "":
            matches = pt.search_nodes(name=a)
            node = matches[0] if matches else None
        else:
            node = pt.get_common_ancestor([a, str(b)])
        if node is not None:
            regimes[node] = chosen
    return regimes, 2


def effective_lengths():
    """Per-gene effective length (mean 1), derived from counts and FPKM.

    For Smart-seq2/RSEM, ``FPKM = count / (L_kb * libsize/1e6)``, so
    ``count/FPKM = L_g * libsize_i/1e6``: a per-gene length times a per-cell
    factor. We recover ``L_g`` by a two-way median decomposition in log space
    (remove the per-cell effect, then take the per-gene median). The absolute
    scale is irrelevant for the Poisson offset, so it is normalized to mean 1.
    Genes with no usable (count>0, FPKM>0) cell get length 1 (no correction).
    """
    if "L" in _EFFLEN_CACHE:
        return _EFFLEN_CACHE["L"]
    cnt = pd.read_csv(COUNTS, sep="\t", index_col=0)
    fpkm = pd.read_csv(FPKM, sep="\t", index_col=0).reindex(
        index=cnt.index, columns=cnt.columns)
    C, F = cnt.values, fpkm.values
    import warnings
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN gene/cell slices
        logr = np.where((C > 0) & (F > 0), np.log(C / F), np.nan)
        a = np.nanmedian(logr, axis=1)                  # per-cell factor
        logL = np.nanmedian(logr - a[:, None], axis=0)  # per-gene (log) length
    L = np.exp(logL)
    L = L / np.nanmean(L)
    L[~np.isfinite(L)] = 1.0
    out = pd.Series(L, index=cnt.columns)
    _EFFLEN_CACHE["L"] = out
    return out


def load_counts(genes=None, min_sublines=None, length_offset=False):
    """Per-cell raw counts for the QC-passed cells, with subline labels.

    Returns ``(counts, genes, clone, offsets)``:
      counts : (n_cells, n_genes) int array (RSEM expected counts, rounded);
      genes  : gene ids; clone : per-cell subline label (tree leaves only);
      offsets : the Poisson offset. By default the per-cell library-size factor
        ``S_i`` (mean 1), shape (n_cells,). With ``length_offset=True`` it is the
        effective-length-scaled offset ``S_i * L_g`` (mean-1 lengths), shape
        (n_cells, n_genes) -- the tximport-style length offset for full-length
        Smart-seq2. Because ``L_g`` is a per-gene constant it is absorbed by the
        per-gene BM/OU baseline, so this does not change the fitted ``K`` (see
        ``length_invariance.py``); it only makes the per-gene mean interpretable
        as length-normalized expression.

    Only cells whose subline is a leaf of the tree are kept (drops C2).
    """
    counts = pd.read_csv(COUNTS, sep="\t", index_col=0)
    c2c = _cell_to_clone()
    leaves = set(tree_leaves())
    keep = [c for c in counts.index if c2c.get(c) in leaves]
    counts = counts.loc[keep]
    clone = np.array([c2c[c] for c in keep])
    X = np.rint(counts.values).astype(float)          # RSEM expected counts -> ints
    if genes is not None:
        cols = [counts.columns.get_loc(g) for g in genes]
        X = X[:, cols]
        gene_ids = np.asarray(genes)
    else:
        gene_ids = counts.columns.to_numpy()
    lib = X.sum(axis=1)
    size_factors = lib / np.mean(lib)                 # per-cell S_i, mean 1
    if not length_offset:
        return X, gene_ids, clone, size_factors
    L = effective_lengths().reindex(gene_ids).values  # per-gene length, mean ~1
    offsets = size_factors[:, None] * L[None, :]       # S_i * L_g, (n_cells, n_genes)
    return X, gene_ids, clone, offsets


def load_subclone_counts(genes=None):
    """Pure-Poisson pseudobulk: counts and offsets *summed* over each subline's cells.

    For cells that share a subline latent exactly, the Poisson sufficient
    statistics collapse to per-subline summed counts and summed size factors --
    so this is the correct *pure-Poisson* (no within-clone plasticity) input to
    ``PoissonObservation``. Rows are aligned to the tree leaf order.

    Returns ``(Y, S, genes, leaves)`` with Y (n_leaves, n_genes) summed counts and
    S (n_leaves,) summed offsets.
    """
    X, gene_ids, clone, sf = load_counts(genes=genes)
    df = pd.DataFrame(X)
    df["__c"] = clone
    Y = df.groupby("__c").sum()
    s = pd.Series(sf).groupby(clone).sum()
    leaves = [l for l in tree_leaves() if l in Y.index]
    Y = Y.reindex(leaves)
    s = s.reindex(leaves)
    return Y.values, s.values, gene_ids, leaves


if __name__ == "__main__":
    X, g, clone, sf = load_counts()
    print(f"per-cell counts: {X.shape[0]} cells x {X.shape[1]} genes; "
          f"size factors in [{sf.min():.2f}, {sf.max():.2f}]")
    pb = load_pseudobulk()
    print(f"tree leaves (sublines): {len(tree_leaves())}")
    print(f"pseudobulk table: {pb.shape[0]} sublines x {pb.shape[1]} genes "
          f"(expressed in all sublines)")
    print(pb.iloc[:5, :3])
