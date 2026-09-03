"""Bridge scPhyTr's ete3-backed :class:`scphytr.utils.tree.Tree` to the array tree used here.

The sampler in :mod:`scphytr.modes.model` needs O(1) access to parent indices, child lists,
pre-order, node depths and per-clade leaf sets, on every one of thousands of Gibbs sweeps.
An ete3 traversal per access is far too slow for that, so this subpackage keeps its own
array-backed :class:`~scphytr.modes._tree.Tree` and converts once, here.

That is a deliberate duplication of tree representations inside one package: the sampler and its
tests were validated against the array form, and rewriting them against ete3 would put correctness
at risk to remove an internal class nobody outside this subpackage sees. Recorded rather than
hidden -- if the array form later earns its place elsewhere, the two should be unified.
"""
from __future__ import annotations

import numpy as np

from ._tree import Tree as ArrayTree

__all__ = ["to_array_tree", "leaf_order"]


def to_array_tree(tree, default_length=1.0):
    """Convert a scPhyTr ``Tree`` (or a bare ete3 node) to the array tree used in this subpackage.

    Returns ``(array_tree, leaf_names)``; ``leaf_names`` is the tip order of the array tree, which
    is what every matrix in this subpackage is indexed by.
    """
    root = getattr(tree, "phylotree", tree)
    root = getattr(root, "get_tree_root", lambda: root)()

    order, parent, dist, name = [], [], [], []
    index = {}
    stack = [(root, -1)]
    while stack:                                     # pre-order, so parents precede children
        node, pidx = stack.pop()
        i = len(order)
        index[id(node)] = i
        order.append(node)
        parent.append(pidx)
        d = getattr(node, "dist", default_length)
        dist.append(default_length if (pidx < 0 or d is None) else float(d))
        nm = getattr(node, "name", "") or f"n{i}"
        name.append(nm)
        for c in reversed(list(node.children)):
            stack.append((c, i))

    children = [[] for _ in order]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)

    at = ArrayTree(np.array(parent), np.array(dist, dtype=float), name, children)
    return at, [at.name[l] for l in at.leaves]


def leaf_order(array_tree):
    """Tip names in the order every matrix in this subpackage is indexed by."""
    return [array_tree.name[l] for l in array_tree.leaves]
