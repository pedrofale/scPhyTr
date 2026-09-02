"""Minimal plotting helpers (matplotlib only, no ete3/ggtree)."""
from __future__ import annotations

import numpy as np

__all__ = ["tree_layout", "plot_tree", "plot_confusion"]


def tree_layout(tree):
    """(x, y) for every node: x = root-to-node distance, y = mean of descendant leaf positions."""
    y = np.zeros(tree.n_nodes)
    pos = {int(l): i for i, l in enumerate(tree.leaves)}
    for v in tree.postorder:
        if tree.is_leaf[v]:
            y[v] = pos[int(v)]
        else:
            y[v] = np.mean([y[c] for c in tree.children[v]])
    return tree.depth.copy(), y


def plot_tree(tree, ax, highlight=None, leaf_values=None, cmap="coolwarm",
              color="0.35", lw=0.8, highlight_color="#d62728", highlight_lw=2.6,
              leaf_size=14):
    """Draw a rectangular cladogram. ``highlight`` = branch (child-node) indices to emphasise."""
    x, y = tree_layout(tree)
    hl = set(int(h) for h in (highlight or []))
    for v in range(tree.n_nodes):
        p = tree.parent[v]
        if p < 0:
            continue
        on = v in hl
        c = highlight_color if on else color
        w = highlight_lw if on else lw
        ax.plot([x[p], x[v]], [y[v], y[v]], color=c, lw=w, solid_capstyle="butt", zorder=3 if on else 1)
    for v in range(tree.n_nodes):                       # vertical connectors
        kids = tree.children[v]
        if len(kids) > 1:
            ax.plot([x[v], x[v]], [min(y[k] for k in kids), max(y[k] for k in kids)],
                    color=color, lw=lw, zorder=1)
    if leaf_values is not None:
        lv = np.asarray(leaf_values, dtype=float)
        ax.scatter(x[tree.leaves], y[tree.leaves], c=lv, cmap=cmap, s=leaf_size,
                   zorder=4, edgecolors="none")
    ax.set_yticks([]); ax.set_xlabel("time (root -> tip)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return ax


def plot_confusion(truth, called, labels, ax, cmap="Blues", title=None):
    """Confusion matrix with counts annotated; rows = truth, columns = calls."""
    truth = np.asarray(truth, dtype=object); called = np.asarray(called, dtype=object)
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for i, t in enumerate(labels):
        for j, c in enumerate(labels):
            M[i, j] = int(np.sum((truth == t) & (called == c)))
    ax.imshow(M, cmap=cmap)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.6 else "black", fontsize=11)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("called"); ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    return M
