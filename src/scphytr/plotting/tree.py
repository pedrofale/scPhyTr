"""Phylogeny plotting: draw the tree and colour branches/clades by a per-node value.

The headline is :func:`rate_tree`, which colours each clade by the evolutionary rate that
:func:`scphytr.tools.detect_rate_shifts` assigned it and marks the branches where a rate
shift was placed --- a direct visual of "which clades evolve faster".
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


def _root_of(tree):
    """Accept a scphytr Tree, an ete3 TreeNode, or anything with a .root/.phylotree."""
    if hasattr(tree, "root") and tree.root is not None:
        return tree.root
    if hasattr(tree, "phylotree") and tree.phylotree is not None:
        return tree.phylotree
    return tree


def _layout(root):
    """Per-node x (root-to-node depth) and y (leaf order; internal = mean of children)."""
    x = {}
    for nd in root.traverse("preorder"):
        x[nd] = (0.0 if nd.up is None else x[nd.up]) + float(nd.dist)
    y = {lf: float(i) for i, lf in enumerate(root.get_leaves())}
    for nd in root.traverse("postorder"):
        if not nd.is_leaf():
            y[nd] = float(np.mean([y[c] for c in nd.children]))
    return x, y


def plot_tree(tree, node_values=None, ax=None, cmap="viridis", vmin=None, vmax=None,
              label_leaves=True, linewidth=2.2, cbar_label="value", title=None):
    """Draw a phylogram, colouring the branch above each node by ``node_values[node]``.

    Parameters
    ----------
    tree : scphytr ``Tree`` / ete3 node.
    node_values : dict {node -> float}, optional. Branches without a value are drawn grey.
    Returns the matplotlib ``Axes``.
    """
    root = _root_of(tree)
    x, y = _layout(root)
    leaves = root.get_leaves()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, max(3.0, 0.22 * len(leaves))))
    cm = plt.get_cmap(cmap)

    vals = node_values or {}
    norm = None
    if vals:
        arr = np.fromiter(vals.values(), float)
        lo = float(np.min(arr)) if vmin is None else vmin
        hi = float(np.max(arr)) if vmax is None else vmax
        norm = mcolors.Normalize(lo, hi if hi > lo else lo + 1e-9)

    hsegs, hcolors, vsegs = [], [], []
    for nd in root.traverse():
        if nd.up is not None:
            hsegs.append([(x[nd.up], y[nd]), (x[nd], y[nd])])
            hcolors.append(cm(norm(vals[nd])) if (norm and nd in vals) else (0.35, 0.35, 0.35, 1))
        if not nd.is_leaf():
            ys = [y[c] for c in nd.children]
            vsegs.append([(x[nd], min(ys)), (x[nd], max(ys))])
    ax.add_collection(LineCollection(vsegs, colors=(0.35, 0.35, 0.35, 1),
                                     linewidths=linewidth * 0.8, zorder=1))
    ax.add_collection(LineCollection(hsegs, colors=hcolors, linewidths=linewidth, zorder=2))

    xmax = max(x.values()) or 1.0
    if label_leaves:
        for lf in leaves:
            ax.text(x[lf] + 0.01 * xmax, y[lf], getattr(lf, "name", ""), va="center", fontsize=7)
    ax.set_xlim(0, xmax * 1.2); ax.set_ylim(-1, len(leaves))
    ax.set_yticks([]); ax.set_xlabel("evolutionary time  (root → tip)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, shrink=0.6, label=cbar_label, pad=0.02)
    if title:
        ax.set_title(title)
    return ax


def rate_tree(tree, shifts, ax=None, cmap="coolwarm", log=True, mark_shifts=True,
              annotate=True, title=None, **kwargs):
    """Colour the phylogeny by each clade's evolutionary rate from ``detect_rate_shifts``.

    Parameters
    ----------
    tree : the same tree passed to :func:`scphytr.tools.detect_rate_shifts`.
    shifts : the dict it returned (keys ``regimes``, ``fit``, ``shifts``).
    log : colour by ``log10`` rate (recommended; rates span orders of magnitude).
    mark_shifts : star the branch where each rate shift begins.
    """
    root = _root_of(tree)
    regimes = shifts["regimes"]
    rates = list(shifts["fit"].params["rates"])
    node_rate = {nd: rates[regimes[nd]] for nd in regimes}
    vals = {nd: (np.log10(max(r, 1e-12)) if log else r) for nd, r in node_rate.items()}
    if title is None:
        n = len(shifts["shifts"])
        title = f"{n} rate shift{'s' if n != 1 else ''} detected" if n else "homogeneous rate (no shift)"
    ax = plot_tree(tree, node_values=vals, ax=ax, cmap=cmap,
                   cbar_label=("log₁₀ rate $\\sigma^2$" if log else "rate $\\sigma^2$"),
                   title=title, **kwargs)
    if (mark_shifts or annotate) and shifts["shifts"]:
        x, y = _layout(root)
        for nd in shifts["shifts"]:
            xb = x[nd.up] if nd.up is not None else x[nd]
            if mark_shifts:
                ax.plot(xb, y[nd], marker="*", color="black", markersize=15, zorder=5)
            if annotate:
                ax.annotate(f"×{node_rate[nd] / rates[0]:.1f}", (xb, y[nd]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8, zorder=6)
    return ax
