"""Read SCOUT's own data format, so we can run our model directly on their files.

SCOUT expects a Newick tree plus a CSV whose rows are cells and whose columns are:
  * ``species``  -- the leaf label, matching a tip in the tree,
  * one column per regime hypothesis (e.g. ``OUM``, ``OU4``) holding the leaf's regime label,
  * every remaining column a gene.

In their shipped simulations the gene NAMES encode the ground-truth model (``BM1_*``, ``OU1_*``,
``OUM_*``), which makes their example data an immediate, honest test set for our classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scphytr.modes._tree import Tree

__all__ = ["load_scout_dataset", "truth_from_gene_names", "write_scout_dataset", "write_newick"]

_REGIME_HINTS = ("OUM", "OUX", "OU2", "OU3", "OU4", "OU5", "regime")


def load_scout_dataset(counts_csv, tree_nwk, species_key="species", regime_key=None,
                       scale_height=1.0, log1p=False):
    """Load a SCOUT counts CSV + Newick tree, aligned to tree tip order.

    Returns ``dict(Y, genes, tree, leaf_regime, cells)`` where ``Y`` is (n_leaves, n_genes) with rows
    in the tree's tip order. Cells whose ``species`` is absent from the tree are dropped; if several
    cells map to one tip they are averaged (SCOUT's simulated files are one cell per tip).
    """
    df = pd.read_csv(counts_csv)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if species_key not in df.columns:
        raise KeyError(f"'{species_key}' column not found; have {list(df.columns)[:8]}...")

    if regime_key is None:
        for c in df.columns:
            if str(c) in _REGIME_HINTS:
                regime_key = str(c)
                break
    meta = {species_key} | ({regime_key} if regime_key else set())
    genes = [c for c in df.columns if c not in meta]

    tree = Tree.from_newick(open(tree_nwk).read())
    if scale_height:
        tree.scale_height(scale_height)
    tip_names = [tree.name[l] for l in tree.leaves]
    pos = {nm: i for i, nm in enumerate(tip_names)}

    df["_row"] = df[species_key].astype(str).map(pos)
    keep = df["_row"].notna()
    if not keep.all():
        df = df[keep]
    df["_row"] = df["_row"].astype(int)

    Y = np.full((len(tip_names), len(genes)), np.nan)
    grouped = df.groupby("_row")[genes].mean()
    Y[grouped.index.values] = grouped.values
    if np.isnan(Y).any():
        missing = int(np.isnan(Y[:, 0]).sum())
        raise ValueError(f"{missing} tips have no expression row")
    if log1p:
        Y = np.log1p(Y)

    leaf_regime = None
    if regime_key:
        lr = df.groupby("_row")[regime_key].first()
        leaf_regime = np.array([lr.loc[i] for i in range(len(tip_names))], dtype=object)

    return dict(Y=Y, genes=list(genes), tree=tree, leaf_regime=leaf_regime, cells=tip_names,
                regime_key=regime_key)


def truth_from_gene_names(genes):
    """SCOUT's simulated gene names encode the true model: ``BM1_3`` -> ``BM1``."""
    out = []
    for g in genes:
        s = str(g)
        base = s.rsplit("_", 1)[0] if "_" in s else s
        out.append("OUX" if base in ("OUM", "OUX") else base)
    return np.array(out, dtype=object)


def write_scout_dataset(path_prefix, Y, tree, leaf_regime=None, gene_names=None,
                        regime_key="OUM", species_key="species", counts=True):
    """Write (Y, tree) in SCOUT's expected input format: ``<prefix>_counts.csv`` + ``<prefix>_tree.nwk``.

    This lets us run the ORIGINAL SCOUT R package on our own simulations. ``Y`` is (n_leaves,
    n_genes) in tree tip order. If ``leaf_regime`` is given it becomes the regime column that SCOUT's
    OUx hypothesis reads; otherwise only BM1/OU1 can be tested.
    """
    import pandas as pd
    Y = np.asarray(Y, dtype=float)
    tips = [tree.name[l] for l in tree.leaves]
    if gene_names is None:
        gene_names = [f"g{i}" for i in range(Y.shape[1])]
    df = pd.DataFrame(Y, columns=list(gene_names))
    if leaf_regime is not None:
        df[regime_key] = list(leaf_regime)
    df[species_key] = tips
    df.index = tips                                  # SCOUT reads with row.names=1
    counts_path = f"{path_prefix}_counts.csv"
    tree_path = f"{path_prefix}_tree.nwk"
    df.to_csv(counts_path)
    write_newick(tree, tree_path)
    return counts_path, tree_path


def write_newick(tree, path=None):
    """Serialise a :class:`scphytr.modes._tree.Tree` back to Newick (names + branch lengths)."""
    def rec(v):
        if tree.is_leaf[v]:
            s = tree.name[v]
        else:
            s = "(" + ",".join(rec(c) for c in tree.children[v]) + ")"
            if tree.name[v]:
                s += tree.name[v]
        if tree.parent[v] >= 0:
            s += f":{tree.dist[v]:.10g}"
        return s
    txt = rec(0) + ";"
    if path:
        with open(path, "w") as fh:
            fh.write(txt + "\n")
    return txt
