"""scPhyTr vs PATH on the PATH paper's own EMT data (PDAC lineage tracing).

PATH (Schiffman et al. 2024) measured the *heritability* of the EMT pseudotime on
single-cell lineage trees via phylogenetic autocorrelation (Moran's I). scPhyTr
models the same continuous EMT trait with one coherent BM/OU process, which yields
*both* PATH's readouts at once:
  * heritability / phylogenetic signal  <- BM-on-tree vs BM-on-star marginal gap;
  * plasticity / transition rate        <- OU mean-reversion alpha (high = fast
                                           mixing = plastic; PATH's transition rate).
Here we load Mouse1.Clone1 (exported from the PATHpaper treedata RDS), subsample
to a common set of cells, and compare scPhyTr's signal/alpha to PATH's Moran's I.
"""
import os
import numpy as np
import pandas as pd
from ete3 import TreeNode

from scphytr.utils.tree import Tree
from scphytr.tools.model_selection import fit_bm, fit_ou
from analysis.benchmark.path_morans import phylo_weights, morans_I
from analysis.benchmark.sim_heritability import star_tree

_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                    "data", "external", "PATHpaper", "EMT_export")


def load_clone(prefix="m1c1", trait="pseudotime"):
    edges = pd.read_csv(os.path.join(_DIR, f"{prefix}_edges.csv"))
    tips = pd.read_csv(os.path.join(_DIR, f"{prefix}_tips.csv"))
    cell = pd.read_csv(os.path.join(_DIR, f"{prefix}_celldata.csv"))
    if trait not in cell.columns:
        raise KeyError(f"{prefix}: trait '{trait}' not available")
    cell = cell.rename(columns={trait: "pseudotime"})
    cell = cell.dropna(subset=["pseudotime"]).groupby("node", as_index=False).first()
    # build ete3 tree from the edge list (node ids; tips have id <= ntip)
    nodes = {}
    def get(i):
        if i not in nodes:
            nodes[i] = TreeNode(); nodes[i].add_feature("nid", i)
        return nodes[i]
    for _, r in edges.iterrows():
        ch = get(int(r["child"])); pa = get(int(r["parent"]))
        ch.dist = float(r["length"]); pa.add_child(ch)
    children = set(edges["child"]); root_id = int(set(edges["parent"]).difference(children).pop())
    label = dict(zip(tips["tip_index"], tips["label"]))
    for i, nd in nodes.items():
        if i in label:
            nd.name = f"{label[i]}__{i}"                      # unique leaf name
    pseudo = dict(zip(cell["node"], cell["pseudotime"]))      # node-id -> EMT
    tree = Tree(); tree.phylotree = nodes[root_id]; tree.root = nodes[root_id]
    # per-leaf EMT, keep leaves with a value
    leaf_emt = {nd.name: float(pseudo[nd.nid]) for nd in tree.phylotree.get_leaves()
                if nd.nid in pseudo}
    return tree, leaf_emt


def subsample(tree, leaf_emt, n, seed=0):
    rng = np.random.default_rng(seed)
    names = list(leaf_emt)
    keep = set(rng.choice(names, size=min(n, len(names)), replace=False))
    keep_nodes = [l for l in tree.phylotree.get_leaves() if l.name in keep]
    tree.phylotree.prune(keep_nodes, preserve_branch_length=True)   # node objects
    pt = tree.phylotree
    # floor zero-length branches (lineage trees have many) so BM/OU pruning is finite
    H = pt.get_farthest_leaf()[1] or 1.0
    eps = 1e-3 * H
    for nd in pt.traverse():
        if nd.up is not None and nd.dist < eps:
            nd.dist = eps
    sub = Tree(); sub.phylotree = pt; sub.root = pt
    vals = {k: leaf_emt[k] for k in pt.get_leaf_names()}
    return sub, vals


def main(n_sub=1200, seed=0):
    from analysis.benchmark.path_morans import path_test
    from scphytr.tools.heritability import pagels_lambda, shared_ancestry_cov
    tree, leaf_emt = load_clone("m1c1")
    print(f"Mouse1.Clone1: {len(leaf_emt)} cells with EMT pseudotime")
    sub, vals = subsample(tree, leaf_emt, n_sub, seed)
    names = sub.phylotree.get_leaf_names()
    x = np.array([vals[n] for n in names])
    H = float(sub.root.get_farthest_leaf()[1]) + float(sub.root.dist)

    # PATH: phylogenetic autocorrelation of EMT pseudotime (+ perm p)
    W = phylo_weights(sub, names)
    I, p_path = path_test(x, W, n_perm=499, rng=np.random.default_rng(seed))
    # scPhyTr: Pagel's lambda heritability (does NOT saturate) + OU transition rate
    C = shared_ancestry_cov(sub)
    lam = pagels_lambda(sub, vals, C=C)
    ou = fit_ou(sub, vals); aH = ou.params["alpha"] * H
    print(f"\nsubsample: {len(names)} cells (tree height H={H:.2f})")
    print(f"  PATH    Moran's I       = {I:+.3f} (perm p={p_path:.3f})")
    print(f"  scPhyTr Pagel lambda    = {lam['lambda']:.3f} (LR p={lam['p']:.1e})  "
          f"<- heritability, analogue of Moran's I")
    print(f"  scPhyTr OU alpha*H      = {aH:.2f}  <- transition rate (same one model)")
    agree = (I > 0 and p_path < 0.05) == (lam["lambda"] > 0.05 and lam["p"] < 0.05)
    print(f"\n  Heritability call -- PATH: {'signal' if p_path<0.05 else 'none'}; "
          f"scPhyTr lambda: {'signal' if lam['p']<0.05 else 'none'}  "
          f"=> {'CONCORDANT' if agree else 'discordant'}")
    print("  scPhyTr gives PATH's heritability (lambda) AND the transition rate (alpha) "
          "from one coherent model.")


if __name__ == "__main__":
    main()
