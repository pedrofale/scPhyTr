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


def spatial_neighbors(adata, spatial_key="spatial", n_neighbors=6, ridge=0.1,
                      key_added="spatial_graph"):
    """Build a leaf-level spatial GMRF graph from per-cell coordinates (``obsm[spatial_key]``).

    Each leaf's coordinate is the mean of its cells'; a symmetric k-nearest-neighbour graph over
    leaves is weighted by a Gaussian kernel of distance, and a sparse precision
    ``Q = (D - W) + ridge·I`` (intrinsic-CAR Laplacian plus a ridge) is stored for the spatial
    random-effect of the phylo⊕space decomposition. ``setup_anndata`` must have been called.
    Stores ``adata.uns[key_added]`` = ``{leaves, coords, weights (CSR), precision (CSR)}``.
    """
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] (cell coordinates) is required.")
    leaves = adata.uns["tree"].phylotree.get_leaf_names()
    pos = {l: i for i, l in enumerate(leaves)}
    idx = np.asarray(adata.obs["_leaf_index"].values, dtype=int)
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    nL, dim = len(leaves), coords.shape[1]
    leaf_xy = np.zeros((nL, dim)); cnt = np.zeros(nL)
    np.add.at(leaf_xy, idx, coords); np.add.at(cnt, idx, 1.0)
    leaf_xy /= np.maximum(cnt[:, None], 1.0)

    k = min(n_neighbors, nL - 1)
    d, nn = cKDTree(leaf_xy).query(leaf_xy, k=k + 1)
    scale = np.median(d[:, 1:]) + 1e-9
    rows, cols, vals = [], [], []
    for i in range(nL):
        for j, dist in zip(nn[i, 1:], d[i, 1:]):
            w = float(np.exp(-(dist * dist) / (2.0 * scale * scale)))
            rows += [i, j]; cols += [j, i]; vals += [w, w]      # symmetrize
    W = sp.csr_matrix((vals, (rows, cols)), shape=(nL, nL))
    W = W.maximum(W.T)
    deg = np.asarray(W.sum(1)).ravel()
    Q = (sp.diags(deg) - W) + ridge * sp.eye(nL)
    adata.uns[key_added] = {"leaves": list(leaves), "coords": leaf_xy,
                            "weights": W.tocsr(), "precision": Q.tocsr()}
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
