"""Phylogeny plotting: draw the tree and colour branches/clades by a per-node value.

Three things live here:

* :func:`plot_tree` --- the phylogram. Colours branches by a continuous ``node_values``, or by a
  categorical ``regimes`` painting (which regime each branch sits in).
* :func:`rate_tree` --- colours each clade by the evolutionary rate that
  :func:`scphytr.tools.detect_rate_shifts` assigned it and marks the branches where a rate shift
  was placed: a direct visual of "which clades evolve faster".
* :func:`expression_tree` --- the tree beside one colour strip per gene or ``obs`` column, for
  reading several variables off the same phylogeny at once. The strips are drawn by
  ``cassiopeia.pl.plot_matplotlib``; we do not reimplement them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib import colors as mcolors

#: branches whose descendant tips span more than one regime, and branches with no value
_GREY = (0.35, 0.35, 0.35, 1.0)
_MIXED = (0.75, 0.75, 0.75, 1.0)


def _root_of(tree):
    """Accept a scphytr Tree, an ete3 TreeNode, the array tree from ``scphytr.modes``, or anything
    with a ``.root`` / ``.phylotree``."""
    if hasattr(tree, "to_newick") and hasattr(tree, "n_nodes"):     # scphytr.modes._tree.Tree
        from ete3 import Tree as _EteTree
        nwk = tree.to_newick()
        for fmt in (1, 0, 100):                 # 1 keeps internal names; fall back if it parses badly
            try:
                return _EteTree(nwk, format=fmt)
            except Exception:
                continue
        raise ValueError("could not parse the array tree's newick with ete3")
    if hasattr(tree, "root") and tree.root is not None:
        return tree.root
    if hasattr(tree, "phylotree") and tree.phylotree is not None:
        return tree.phylotree
    return tree


def _newick_of(tree):
    """Newick text for anything :func:`_root_of` accepts."""
    if hasattr(tree, "to_newick"):
        return tree.to_newick()
    return _root_of(tree).write(format=1, format_root_node=True)


def paint_from_leaves(tree, leaf_regime):
    """Regime of every node, or ``None`` where its descendant tips span more than one.

    The ete3 counterpart of :func:`scphytr.modes.baseline.paint_regimes` with ``mixed=None``: a
    node takes the unique regime of its descendant tips, and a branch above a regime split is
    given to no regime rather than attributed to one of them.

    ``leaf_regime`` is either a ``{tip name -> label}`` mapping or a sequence in the tree's own
    tip order. Returns ``{node -> label or None}``.
    """
    root = _root_of(tree)
    leaves = root.get_leaves()
    if hasattr(leaf_regime, "get") and not isinstance(leaf_regime, (list, tuple, np.ndarray)):
        lab = {lf: leaf_regime[lf.name] for lf in leaves}
    else:
        seq = list(leaf_regime)
        if len(seq) != len(leaves):
            raise ValueError(f"leaf_regime has {len(seq)} entries for {len(leaves)} tips")
        lab = dict(zip(leaves, seq))
    seen = {}
    for nd in root.traverse("postorder"):
        seen[nd] = {lab[nd]} if nd.is_leaf() else set().union(*(seen[c] for c in nd.children))
    return {nd: (next(iter(v)) if len(v) == 1 else None) for nd, v in seen.items()}


def regime_palette(regimes, cmap="tab10"):
    """Stable ``{label -> colour}``, in first-seen order.

    Pass the union of several panels' regimes to keep one label the same colour in all of them.
    """
    labels = list(dict.fromkeys(regimes))
    cm = plt.get_cmap(cmap)
    n = int(getattr(cm, "N", 256))
    if n >= 64:                                  # a continuous map: spread the labels over it
        return {r: cm(i / max(len(labels) - 1, 1)) for i, r in enumerate(labels)}
    return {r: cm(i % n) for i, r in enumerate(labels)}


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


def _adata_leaf_values(adata, color):
    """Per-leaf value for ``color``: a gene in ``var_names`` (mean log1p expression over the tip's
    cells), or an ``obs`` column -- numeric ones averaged, categorical ones taking the commonest
    label at the tip."""
    import pandas as pd
    sp = np.asarray(adata.obs[adata.uns.get("_species_obs", "species")]).astype(str)
    if color in adata.var_names:
        X = adata[:, color].X
        v = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(float).ravel()
        sf = np.asarray(adata.obs.get(adata.uns.get("_size_factor_obs", "size_factors"), 1.0), float)
        v = np.log1p(v / np.maximum(sf, 1e-9))
    elif color in adata.obs:
        col = adata.obs[color]
        if not pd.api.types.is_numeric_dtype(col):
            return (pd.Series(np.asarray(col).astype(str))
                      .groupby(sp).agg(lambda t: t.value_counts().index[0]).to_dict())
        v = np.asarray(col.values, float)
    else:
        raise KeyError(f"color '{color}' is not a gene (var_names) or obs column")
    return pd.Series(v).groupby(sp).mean().to_dict()


def plot_tree(tree, node_values=None, color=None, ax=None, cmap="viridis", vmin=None, vmax=None,
              label_leaves=True, linewidth=2.2, cbar_label="value", title=None,
              regimes=None, palette=None, regime_cmap="tab10", legend=True, tip_strip=False):
    """Draw a phylogram, colouring the branch above each node by ``node_values[node]``.

    Parameters
    ----------
    tree : scphytr ``Tree`` / ete3 node / the array tree from ``scphytr.modes``, or an ``AnnData``
        (then the tree is ``uns['tree']``).
    node_values : dict {node -> float}, optional. Branches without a value are drawn grey.
    color : when ``tree`` is an AnnData, a gene name or obs column to colour leaves by.
    regimes : per-tip regime labels -- a ``{tip name -> label}`` mapping or a sequence in tip order.
        Paints each branch with the regime of its descendant tips (:func:`paint_from_leaves`);
        branches above a regime split stay grey rather than being attributed to one side. Mutually
        exclusive with ``node_values`` / ``color``, which are the continuous read-out.
    palette : ``{label -> colour}`` for ``regimes``; built with :func:`regime_palette` when absent.
        Pass one explicitly to keep a label the same colour across several panels.
    regime_cmap : colormap the palette is drawn from when ``palette`` is not given. Separate from
        ``cmap``, which belongs to the continuous read-out and is unused when ``regimes`` is set.
    tip_strip : draw a column of tip colours past the tips. On a large tree the branches alone are
        too fine to read the blocks off, so this is what actually shows the painting.

    Returns the matplotlib ``Axes``.
    """
    if regimes is not None and (node_values is not None or color is not None):
        raise ValueError("pass either regimes= (categorical) or node_values=/color= (continuous)")
    if hasattr(tree, "uns"):                                   # AnnData
        adata = tree
        leaf_vals = _adata_leaf_values(adata, color) if color is not None else {}
        tree = adata.uns["tree"]
        root = _root_of(tree)
        node_values = {lf: leaf_vals[lf.name] for lf in root.get_leaves() if lf.name in leaf_vals}
        if title is None and color is not None:
            title = str(color)
        cbar_label = str(color) if color is not None else cbar_label
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

    painted = pal = None
    if regimes is not None:
        painted = paint_from_leaves(root, regimes)
        labels = [painted[lf] for lf in leaves]
        pal = dict(palette) if palette is not None else regime_palette(labels, cmap=regime_cmap)

    hsegs, hcolors, vsegs, vcolors = [], [], [], []
    for nd in root.traverse():
        if nd.up is not None:
            hsegs.append([(x[nd.up], y[nd]), (x[nd], y[nd])])
            if painted is not None:
                hcolors.append(pal.get(painted[nd], _MIXED) if painted[nd] is not None else _MIXED)
            else:
                hcolors.append(cm(norm(vals[nd])) if (norm and nd in vals) else _GREY)
        if not nd.is_leaf():
            ys = [y[c] for c in nd.children]
            vsegs.append([(x[nd], min(ys)), (x[nd], max(ys))])
            # the connector belongs to the regime only when every child agrees on one
            if painted is not None:
                vcolors.append(pal.get(painted[nd], _MIXED) if painted[nd] is not None else _MIXED)
            else:
                vcolors.append(_GREY)
    ax.add_collection(LineCollection(vsegs, colors=vcolors, linewidths=linewidth * 0.8, zorder=1))
    ax.add_collection(LineCollection(hsegs, colors=hcolors, linewidths=linewidth, zorder=2))

    xmax = max(x.values()) or 1.0
    if painted is not None and tip_strip:
        ax.scatter([xmax * 1.06] * len(leaves), [y[lf] for lf in leaves],
                   c=[pal.get(painted[lf], _MIXED) for lf in leaves],
                   s=8, marker="s", linewidths=0, clip_on=False, zorder=4)
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
    if painted is not None and legend:
        import collections as _c
        n = _c.Counter(painted[lf] for lf in leaves)
        ax.legend(handles=[Line2D([], [], color=pal[k], lw=3, label=f"{k}  ({n[k]} tips)")
                           for k in sorted(n, key=lambda k: -n[k])],
                  frameon=False, fontsize=8, loc="lower right", borderaxespad=0.8)
    if title:
        ax.set_title(title)
    return ax


def _resolve_tree_shifts(tree, shifts):
    """Accept (tree, shifts) or an AnnData carrying uns['tree'] + uns['rate_shifts']."""
    if hasattr(tree, "uns") and shifts is None:
        return tree.uns["tree"], tree.uns["rate_shifts"]
    return tree, shifts


def rate_tree(tree, shifts=None, ax=None, cmap="coolwarm", log=True, mark_shifts=True,
              annotate=True, title=None, **kwargs):
    """Colour the phylogeny by each clade's evolutionary rate from ``detect_rate_shifts``.

    Parameters
    ----------
    tree : the same tree passed to :func:`scphytr.tools.detect_rate_shifts`.
    shifts : the dict it returned (keys ``regimes``, ``fit``, ``shifts``).
    log : colour by ``log10`` rate (recommended; rates span orders of magnitude).
    mark_shifts : star the branch where each rate shift begins.
    """
    tree, shifts = _resolve_tree_shifts(tree, shifts)
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


def _label_strips(ax, keys):
    """Name each colour strip.

    ``cassiopeia.pl.plot_matplotlib`` draws the strips as ``Polygon`` patches past the tips, in
    ``meta_data`` order and progressively further from the tree, but leaves them unlabelled --
    which makes a multi-gene panel unreadable. Group the patches by x centre and order them by
    distance from the tree body to recover that order.
    """
    polys = [a for a in ax.get_children() if isinstance(a, Polygon)]
    lines = [a for a in ax.get_children()
             if isinstance(a, Line2D) and len(np.asarray(a.get_xdata()))]
    if not polys or not lines:
        return
    bx = np.concatenate([np.asarray(l.get_xdata(), float) for l in lines])
    lo, hi = float(bx.min()), float(bx.max())
    centres = np.array([p.get_xy()[:, 0].mean() for p in polys])
    outside = centres[(centres > hi) | (centres < lo)]
    if outside.size == 0:
        return
    uniq = np.unique(np.round(outside, 6))
    tip = hi if float(uniq.mean()) > hi else lo
    uniq = uniq[np.argsort(np.abs(uniq - tip))]
    ytop = max(float(p.get_xy()[:, 1].max()) for p in polys)
    y0, y1 = ax.get_ylim()
    for xc, k in zip(uniq, keys):
        ax.text(float(xc), ytop, f" {k}", rotation=90, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(y0, max(y1, ytop + 0.16 * (ytop - min(y0, 0.0) or 1.0)))


def expression_tree(tree, keys, adata=None, cell_meta=None, orient="right", figsize=(8.0, 9.0),
                    continuous_cmap="viridis", categorical_cmap="tab10", add_root=True,
                    label_strips=True, title=None, **kwargs):
    """Draw the tree beside one colour strip per key -- genes, ``obs`` columns, or regime labels.

    This is the read-several-variables-at-once view: :func:`plot_tree` colours branches by one
    thing, while this puts a strip per key alongside the tips, so expression of several genes can
    be compared against each other and against a painting on the same phylogeny.

    The strips themselves are drawn by ``cassiopeia.pl.plot_matplotlib`` -- we do not reimplement
    them. What is added here is resolving ``keys`` against an ``AnnData`` onto the tips, and
    labelling the strips, which cassiopeia leaves off.

    Parameters
    ----------
    tree : anything :func:`plot_tree` accepts, or an ``AnnData`` carrying ``uns['tree']``.
    keys : one key or a list. Genes in ``var_names``, or ``obs`` columns; categorical columns get a
        discrete palette and numeric ones a continuous map, as cassiopeia decides per column.
    adata : where ``keys`` are looked up, when ``tree`` is not itself an ``AnnData``.
    cell_meta : a ready ``{tip name -> value}`` frame, instead of ``adata``. Its columns are used
        verbatim, so this is the way in for values that are not in an ``AnnData`` at all.
    label_strips : name each strip. Only meaningful for ``orient`` "left"/"right"; ignored
        otherwise, since the strips then run along y.

    Requires ``cassiopeia-lineage``, imported lazily -- as in :mod:`scphytr.simulation`, it is an
    optional dependency rather than a hard one.

    Returns the matplotlib ``Axes``.
    """
    import pandas as pd

    keys = [keys] if isinstance(keys, str) else list(keys)
    if hasattr(tree, "uns"):                                   # AnnData
        adata, tree = tree, tree.uns["tree"]
    if cell_meta is None:
        if adata is None:
            raise ValueError("pass adata= (or cell_meta=) to say where the values come from")
        cell_meta = pd.DataFrame({k: pd.Series(_adata_leaf_values(adata, k)) for k in keys})
    else:
        cell_meta = pd.DataFrame(cell_meta)
        missing = [k for k in keys if k not in cell_meta.columns]
        if missing:
            raise KeyError(f"cell_meta has no column(s) {missing}")
        cell_meta = cell_meta[keys]

    tips = [lf.name for lf in _root_of(tree).get_leaves()]
    absent = [t for t in tips if t not in cell_meta.index]
    if absent:
        raise KeyError(f"{len(absent)} tip(s) missing from the values, e.g. {absent[:3]}")

    try:
        from cassiopeia.data import CassiopeiaTree
        from cassiopeia.pl import plot_matplotlib
    except ImportError as e:                                   # pragma: no cover
        raise ImportError("expression_tree needs cassiopeia-lineage: pip install cassiopeia-lineage"
                          ) from e

    ct = CassiopeiaTree(tree=_newick_of(tree), cell_meta=cell_meta.loc[tips])
    fig, ax = plot_matplotlib(ct, meta_data=keys, orient=orient, figsize=figsize,
                              continuous_cmap=continuous_cmap, categorical_cmap=categorical_cmap,
                              add_root=add_root, **kwargs)
    if label_strips and orient in ("left", "right"):
        _label_strips(ax, keys)
    if title:
        ax.set_title(title)
    return ax
