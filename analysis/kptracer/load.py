"""Load a KP-Tracer tumor: phylogeny + log-normalized expression, aligned.

Data layout inside ``KPTracer-Data.tar.gz`` (Zenodo 5847462; Yang*, Jones* et
al., Cell 2022):

  expression/adata_processed.nt.h5ad   integrated sgNT AnnData (obs['Tumor'],
                                       'leiden_sub', UMAP, ...)
  trees/{tumor}_tree.nwk               per-tumor Newick (ete3 format 1)
  trees/{tumor}_character_matrix.txt   lineage-tracing character matrix

Tree leaves are cell barcodes; they map to AnnData rows by name intersection.

The expensive piece is the phylogenetic (shared-time) covariance ``C``: the
generic ``scphytr.utils.covariance.phylo_times`` does O(n^2) ``get_common_ancestor``
calls, which is far too slow for trees with hundreds–thousands of leaves. Here we
build ``C`` in one post-order traversal: order the leaves by traversal so every
node owns a contiguous block, then for each internal node at depth ``d`` set the
cross-child blocks of ``C`` to ``d`` (vectorized). The diagonal is the per-leaf
root-to-tip depth. This is O(n^2) writes but with numpy block assignment, not
per-pair tree queries.
"""

import os

import numpy as np
import ete3


def shared_time_matrix(tree):
    """Return (leaf_names, C) where C[i,j] = shared root-to-MRCA time.

    ``C`` is the Brownian-motion phylogenetic covariance (unit rate). Built in a
    single traversal; leaves are returned in the traversal order used to index C.
    """
    # depth (root-to-node distance) for every node
    depth = {}
    for node in tree.traverse("preorder"):
        if node.is_root():
            depth[node] = float(node.dist)  # include any root branch
        else:
            depth[node] = depth[node.up] + float(node.dist)

    leaves = tree.get_leaves()
    n = len(leaves)
    leaf_index = {leaf: i for i, leaf in enumerate(leaves)}

    C = np.zeros((n, n))
    # diagonal: root-to-tip depth
    for leaf in leaves:
        C[leaf_index[leaf], leaf_index[leaf]] = depth[leaf]

    # post-order: each internal node fills cross-child-block shared times
    cache = {}  # node -> np.array of leaf indices beneath it
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            cache[node] = np.array([leaf_index[node]])
            continue
        child_blocks = [cache[c] for c in node.children]
        d = depth[node]
        for a in range(len(child_blocks)):
            for b in range(a + 1, len(child_blocks)):
                ia, ib = child_blocks[a], child_blocks[b]
                C[np.ix_(ia, ib)] = d
                C[np.ix_(ib, ia)] = d
        cache[node] = np.concatenate(child_blocks)

    names = [leaf.name for leaf in leaves]
    return names, C


def _ensure_branch_lengths(tree):
    """If the tree has no usable branch lengths (all ~0), assign unit branches.

    Cassiopeia trees sometimes carry only topology; BM still needs edge lengths.
    Returns (tree, used_unit) where used_unit flags that we imposed unit branches.
    """
    dists = [float(n.dist) for n in tree.traverse() if not n.is_root()]
    if len(dists) == 0 or np.allclose(dists, 0.0):
        for n in tree.traverse():
            if not n.is_root():
                n.dist = 1.0
        return tree, True
    return tree, False


def load_tumor(tumor, data_dir, adata=None, min_branch=1e-6, log1p=True, n_hvg=None):
    """Load one tumor's tree + log-normalized expression, aligned to leaves.

    Parameters
    ----------
    tumor : str, e.g. "3726_NT_T2".
    data_dir : path to the extracted ``KPTracer-Data`` directory.
    adata : optional preloaded AnnData (avoids re-reading the 1 GB h5ad).
    log1p : if True and the data look like counts, apply CPM-style log1p.
    n_hvg : if set, keep only the top-``n_hvg`` most variable genes (computed on
        the tumor's leaves after any normalization).

    Returns
    -------
    dict with: tumor, tree (ete3), leaf_names (list, C order), C (n,n),
    Y (n, p) expression, genes (p,), obs (DataFrame, leaf order), used_unit_bl.
    """
    import anndata as anndata_mod

    if adata is None:
        adata = anndata_mod.read_h5ad(os.path.join(data_dir, "expression", "adata_processed.nt.h5ad"))

    tree_fp = os.path.join(data_dir, "trees", f"{tumor}_tree.nwk")
    tree = ete3.Tree(tree_fp, format=1)
    tree, used_unit = _ensure_branch_lengths(tree)

    # floor tiny/zero internal branches so C stays positive definite
    for node in tree.traverse():
        if not node.is_root() and float(node.dist) < min_branch:
            node.dist = min_branch

    # keep only leaves that are profiled cells in this tumor
    cells = set(adata.obs_names)
    tumor_cells = set(adata.obs_names[adata.obs["Tumor"] == tumor]) if "Tumor" in adata.obs else cells
    keep = [l.name for l in tree.get_leaves() if l.name in tumor_cells]
    if len(keep) < 10:
        raise ValueError(f"Only {len(keep)} leaves of {tumor} found in AnnData.")
    tree.prune(keep, preserve_branch_length=True)

    names, C = shared_time_matrix(tree)

    sub = adata[names]
    X = sub.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    looks_like_counts = np.allclose(X, np.round(X)) and X.min() >= 0
    if log1p and looks_like_counts:
        lib = X.sum(1, keepdims=True)
        lib[lib == 0] = 1.0
        X = np.log1p(X / lib * np.median(lib))

    genes = np.asarray(sub.var_names)
    if n_hvg is not None and n_hvg < X.shape[1]:
        v = X.var(0)
        hv = np.argsort(v)[::-1][:n_hvg]
        hv.sort()
        X = X[:, hv]
        genes = genes[hv]

    return {
        "tumor": tumor,
        "tree": tree,
        "leaf_names": names,
        "C": C,
        "Y": X,
        "genes": genes,
        "obs": sub.obs.copy(),
        "used_unit_bl": used_unit,
    }
