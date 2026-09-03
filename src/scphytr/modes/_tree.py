"""Minimal array-backed rooted tree + Newick parser.

Deliberately dependency-free (numpy only) so this subproject stands alone: no ete3, no ape, no
scPhyTr. Nodes are stored in a flat array; branch ``b`` is identified with its CHILD node, matching
the model notation where ``theta_b = theta_{parent(b)} + delta_b``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Tree", "parse_newick"]


class Tree:
    """A rooted tree over ``n_nodes`` nodes, root at index 0.

    Attributes
    ----------
    parent : (n_nodes,) int      parent index, -1 at the root
    dist : (n_nodes,) float      length of the branch ABOVE the node (0 at the root)
    name : list[str]             node labels ('' for unnamed internal nodes)
    children : list[list[int]]
    leaves : (n_leaves,) int     node indices of the tips, in Newick order
    preorder, postorder : (n_nodes,) int
    depth : (n_nodes,) float     root-to-node distance
    """

    def __init__(self, parent, dist, name, children):
        self.parent = np.asarray(parent, dtype=int)
        self.dist = np.asarray(dist, dtype=float)
        self.name = list(name)
        self.children = children
        n = len(self.parent)
        self.n_nodes = n
        self.is_leaf = np.array([len(c) == 0 for c in children], dtype=bool)
        self.leaves = np.where(self.is_leaf)[0]
        self.n_leaves = len(self.leaves)
        # pre/post order by iterative DFS (children pushed reversed so preorder is left-to-right)
        pre, stack = [], [0]
        while stack:
            v = stack.pop()
            pre.append(v)
            stack.extend(reversed(self.children[v]))
        self.preorder = np.array(pre, dtype=int)
        self.postorder = self.preorder[::-1].copy()
        d = np.zeros(n)
        for v in self.preorder:
            if self.parent[v] >= 0:
                d[v] = d[self.parent[v]] + self.dist[v]
        self.depth = d
        self.leaf_index = {nm: i for i, nm in enumerate([self.name[l] for l in self.leaves])}

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_newick(cls, text, default_length=1.0):
        return parse_newick(text, default_length=default_length)

    @classmethod
    def balanced(cls, n_leaves, branch_length=1.0):
        """A perfectly balanced binary tree with ``n_leaves`` = 2**k tips (all branches equal)."""
        k = int(np.log2(n_leaves))
        if 2 ** k != n_leaves:
            raise ValueError("n_leaves must be a power of two")
        parent, dist, name, children = [-1], [0.0], [""], [[]]
        frontier = [0]
        for _ in range(k):
            nxt = []
            for p in frontier:
                for _ in range(2):
                    i = len(parent)
                    parent.append(p); dist.append(branch_length); name.append("")
                    children.append([]); children[p].append(i); nxt.append(i)
            frontier = nxt
        for j, v in enumerate(frontier):
            name[v] = f"t{j + 1}"
        return cls(parent, dist, name, children)

    # -- helpers ------------------------------------------------------------
    def scale_height(self, height=1.0):
        """Rescale all branch lengths so the mean root-to-tip distance equals ``height``."""
        cur = float(np.mean(self.depth[self.leaves]))
        if cur <= 0:
            raise ValueError("degenerate tree height")
        self.dist = self.dist * (height / cur)
        d = np.zeros(self.n_nodes)
        for v in self.preorder:
            if self.parent[v] >= 0:
                d[v] = d[self.parent[v]] + self.dist[v]
        self.depth = d
        return self

    def to_newick(self, precision=6):
        """Newick text for this tree, round-tripping with :meth:`from_newick`.

        Written iteratively rather than recursively so a deep unbalanced tree cannot exhaust the
        interpreter's stack. Internal node names are kept -- they are what lets a caller line a
        painting or a per-branch value back up with the tree.
        """
        parts = [None] * self.n_nodes
        for v in self.postorder:
            kids = self.children[v]
            inner = "(" + ",".join(parts[c] for c in kids) + ")" if kids else ""
            parts[v] = f"{inner}{self.name[v] or ''}:{self.dist[v]:.{precision}g}"
        root = int(np.flatnonzero(self.parent < 0)[0])
        return parts[root] + ";"

    def leaf_sets(self):
        """For every node, the array of LEAF-ORDER indices beneath it (postorder accumulation)."""
        pos = {int(l): i for i, l in enumerate(self.leaves)}
        out = [None] * self.n_nodes
        for v in self.postorder:
            if self.is_leaf[v]:
                out[v] = np.array([pos[int(v)]], dtype=int)
            else:
                out[v] = np.concatenate([out[c] for c in self.children[v]])
        return out

    def branch_labels(self):
        """Human-readable label per branch (= per child node)."""
        return [self.name[v] if self.name[v] else f"node{v}" for v in range(self.n_nodes)]

    def __repr__(self):
        return f"Tree(n_nodes={self.n_nodes}, n_leaves={self.n_leaves})"


def parse_newick(text, default_length=1.0):
    """Parse a Newick string into a :class:`Tree`. Missing branch lengths -> ``default_length``."""
    s = text.strip()
    if s.endswith(";"):
        s = s[:-1]
    parent, dist, name, children = [], [], [], []

    def new_node(p):
        i = len(parent)
        parent.append(p); dist.append(np.nan); name.append(""); children.append([])
        if p >= 0:
            children[p].append(i)
        return i

    pos = 0

    def parse_node(p):
        nonlocal pos
        me = new_node(p)
        if pos < len(s) and s[pos] == "(":
            pos += 1
            while True:
                parse_node(me)
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    continue
                if pos < len(s) and s[pos] == ")":
                    pos += 1
                break
        start = pos
        while pos < len(s) and s[pos] not in "(),:;":
            pos += 1
        lbl = s[start:pos].strip().strip("'\"")
        if lbl:
            name[me] = lbl
        if pos < len(s) and s[pos] == ":":
            pos += 1
            start = pos
            while pos < len(s) and (s[pos].isdigit() or s[pos] in ".eE+-"):
                pos += 1
            dist[me] = float(s[start:pos])
        return me

    parse_node(-1)
    dist = [default_length if np.isnan(d) else d for d in dist]
    dist[0] = 0.0
    return Tree(parent, dist, name, children)
