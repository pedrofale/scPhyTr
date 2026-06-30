"""Load real PEtracer (Koblan/Yost/Zheng et al., Science 2025) spatial lineage-tracing tumors into
scPhyTr's (ancestry x space x counts) pipeline.

Data: figshare 10.6084/m9.figshare.28473866 (``M{1,2,3}_tumor_tracing.h5td``). Each ``.h5td`` is a
``treedata.TreeData``: MERFISH counts (``layers['counts']``, 124-gene panel), spatial coordinates
(``obsm['spatial']``), per-cell metadata (clade, tumor-boundary distance, subtype, fitness), and a
set of reconstructed Cassiopeia lineage trees in ``obst`` (one per clonal tumour, plus ``_collapsed``
variants). Leaves are individual cells (one cell per leaf), so there is no pseudobulk -- the
subclonal count model runs with one replicate per leaf.

We convert one tumour's networkx tree to the ete3 tree scPhyTr expects, using node ``time``
(time-calibrated lineage depth) for branch lengths -- the natural Brownian-motion divergence axis --
and return a raw-count AnnData wired for ``pp.setup_anndata`` / ``pp.spatial_neighbors``.
"""
import numpy as np
import anndata as ad
import networkx as nx
from ete3 import Tree as ETree

from scphytr.utils.tree import Tree

DATA = "data/external/petracer"


def _root(g):
    return [n for n in g.nodes if g.in_degree(n) == 0][0]


def nx_to_ete3(g, branch="time", min_len=1e-6):
    """networkx DiGraph -> ete3 tree. ``branch='time'`` uses child.time-parent.time (BM divergence);
    ``branch='length'`` uses the edit-distance edge length. Leaf names are the cell ids."""
    root = _root(g)
    et = ETree(name=str(root)); et.dist = 0.0
    nodemap = {root: et}
    for parent in nx.bfs_tree(g, root):
        pnode = nodemap[parent]
        for child in g.successors(parent):
            if branch == "time":
                d = g.nodes[child].get("time", 0.0) - g.nodes[parent].get("time", 0.0)
            else:
                d = g.edges[parent, child].get("length", 1.0)
            nodemap[child] = pnode.add_child(name=str(child), dist=float(max(d, min_len)))
    return et


def available_trees(path):
    """Map of non-collapsed tree key -> leaf count, largest first."""
    import treedata as td
    t = td.read_h5td(path)
    out = {}
    for k in t.obst_keys():
        if k.endswith("_collapsed"):
            continue
        g = t.obst[k]
        out[k] = sum(1 for n in g.nodes if g.out_degree(n) == 0)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def load_tumor_tree(path, tree_key, branch="time", layer="counts"):
    """Return ``(adata, tree)`` for one clonal tumour, ready for ``pp.setup_anndata``.

    ``adata.X`` = raw MERFISH counts (for the NB/subclonal model); ``obsm['spatial']`` = coords;
    ``obs['species']`` = cell id (= leaf). Carries through useful covariates (clade,
    tumor_boundary_dist, within_tumor, cell_subtype, fitness, volume, total_counts).
    """
    import treedata as td
    t = td.read_h5td(path)
    g = t.obst[tree_key]
    leaves = [n for n in g.nodes if g.out_degree(n) == 0]
    keep = [c for c in leaves if c in set(t.obs_names)]
    sub = t[keep].copy()                                   # cells in tree-leaf order

    X = sub.layers[layer]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    A = ad.AnnData(X=np.asarray(X, dtype=float))
    A.var_names = list(sub.var_names)
    A.obs_names = list(sub.obs_names)
    carry = ["clade", "clone", "tumor", "tumor_boundary_dist", "within_tumor",
             "cell_subtype", "fitness", "volume", "total_counts", "cell_neighborhood"]
    for c in carry:
        if c in sub.obs:
            A.obs[c] = np.asarray(sub.obs[c].values)
    A.obs["species"] = list(A.obs_names)
    A.obsm["spatial"] = np.asarray(sub.obsm["spatial"], dtype=float)

    et = nx_to_ete3(g, branch=branch)
    # prune the ete3 tree to exactly the kept leaves (drop any leaf without expression data)
    if len(keep) != len(leaves):
        et.prune([str(c) for c in keep], preserve_branch_length=True)
    tree = Tree(); tree.phylotree = et; tree.root = et.get_tree_root()
    return A, tree


if __name__ == "__main__":
    import sys
    p = f"{DATA}/M2_tumor_tracing.h5td"
    print("available trees (leaves):", available_trees(p))
    A, tree = load_tumor_tree(p, sys.argv[1] if len(sys.argv) > 1 else "1")
    print("adata:", A.shape, "| leaves:", len(tree.phylotree.get_leaf_names()),
          "| spatial:", A.obsm["spatial"].shape)
