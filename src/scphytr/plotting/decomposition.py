"""Plot the tree⊕space variance decomposition (``pl.variance_decomposition``)."""
import numpy as np


def variance_decomposition(adata, genes=None, label=False, ax=None, s=45, cmap="coolwarm"):
    """Scatter genes on the heritable–niche plane from ``tl.decompose_variance``.

    Plots each gene's spatial (niche) variance vs its phylogenetic (heritable) variance, colored by
    ``frac_heritable``; the ``y = x`` line separates heritable-dominated (upper) from niche-dominated
    (lower) genes. The principled replacement for a descriptive "heritable vs spatially-restricted"
    scatter. Needs ``var['v_phylo','v_space','frac_heritable']`` (run ``tl.decompose_variance`` first).
    """
    import matplotlib.pyplot as plt
    if "frac_heritable" not in adata.var:
        raise KeyError("run tl.decompose_variance first (needs var['v_phylo','v_space','frac_heritable'])")
    genes = list(adata.var_names) if genes is None else list(genes)
    v = adata.var.loc[genes]
    vp = np.asarray(v["v_phylo"], float)
    vs = np.asarray(v["v_space"], float)
    fr = np.asarray(v["frac_heritable"], float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 5.0))
    lim = float(np.nanmax([vp.max(initial=1e-3), vs.max(initial=1e-3)])) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, zorder=1, label="equal (frac=0.5)")
    sc = ax.scatter(vs, vp, c=fr, cmap=cmap, vmin=0, vmax=1, s=s, edgecolor="k",
                    linewidth=0.4, zorder=3)
    if label:
        for g, x, y in zip(genes, vs, vp):
            ax.annotate(g, (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("niche (spatial) variance  $v_{space}$")
    ax.set_ylabel("heritable (tree) variance  $v_{phylo}$")
    ax.set_title("Variance decomposition: heritable vs niche")
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("frac heritable")
    ax.legend(fontsize=8, loc="lower right")
    return ax
