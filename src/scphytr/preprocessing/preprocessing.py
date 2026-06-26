"""Preprocessing (``scphytr.pp``): attach a phylogeny to an AnnData and align cells to leaves."""
import numpy as np


def setup_anndata(adata, tree, species_obs="species", size_factor_obs="size_factors"):
    """Attach ``tree`` to ``adata`` and align cells to its leaves (no pseudobulk).

    Stores the tree in ``adata.uns['tree']`` and, for the subclonal count models, a per-cell
    integer leaf index in ``adata.obs['_leaf_index']`` plus Poisson size factors in
    ``adata.obs[size_factor_obs]`` (library size / its mean) if absent. ``adata.obs[species_obs]``
    must label each cell with its leaf (subclone, or the cell itself on a single-cell tree).
    """
    adata.uns["tree"] = tree
    leaves = tree.phylotree.get_leaf_names()
    pos = {s: i for i, s in enumerate(leaves)}
    if species_obs not in adata.obs:
        raise KeyError(f"adata.obs['{species_obs}'] (cell -> leaf label) is required; "
                       f"set it before setup_anndata.")
    species = np.asarray(adata.obs[species_obs].values).astype(str)
    missing = set(species) - set(pos)
    if missing:
        raise ValueError(f"{len(missing)} cell labels are not tree leaves, e.g. {list(missing)[:3]}")
    adata.obs["_leaf_index"] = np.array([pos[s] for s in species], dtype=int)
    adata.uns["_species_obs"] = species_obs
    if size_factor_obs not in adata.obs:
        X = adata.X
        lib = np.asarray(X.sum(1)).ravel() if hasattr(X, "sum") else np.asarray(X).sum(1)
        lib = np.asarray(lib, dtype=float)
        adata.obs[size_factor_obs] = lib / np.mean(lib[lib > 0])
    adata.uns["_size_factor_obs"] = size_factor_obs
    return adata


def cut_tree(adata, min_cells=10, species_obs="species"):
    """Prune leaves with fewer than ``min_cells`` cells (and drop those cells); floor branches."""
    tree = adata.uns["tree"]
    counts = adata.obs[species_obs].astype(str).value_counts()
    keep = set(counts.index[counts >= min_cells])
    keep_leaves = [l for l in tree.phylotree.get_leaf_names() if l in keep]
    tree.phylotree.prune(keep_leaves, preserve_branch_length=True)
    H = tree.root.get_farthest_leaf()[1] or 1.0
    for nd in tree.phylotree.traverse():
        if nd.up is not None and nd.dist < 1e-3 * H:
            nd.dist = 1e-3 * H
    sub = adata[adata.obs[species_obs].astype(str).isin(keep)].copy()
    setup_anndata(sub, tree, species_obs=species_obs)
    return sub
