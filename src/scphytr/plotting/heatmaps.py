"""Matrix / loadings plots for the AnnData read-outs (``pl.matrix``, ``pl.loadings``)."""
import numpy as np
import matplotlib.pyplot as plt


def matrix(adata, key="K_corr", genes_key=None, ax=None, cmap="RdBu_r",
           vmin=-1, vmax=1, title=None):
    """Heatmap of a gene-gene matrix stored in ``adata.uns`` (e.g. the evolutionary
    correlation from ``tl.evolutionary_correlation``)."""
    M = np.asarray(adata.uns[key])
    genes = adata.uns.get(genes_key or (key.split("_")[0] + "_genes"))
    if ax is None:
        _, ax = plt.subplots(figsize=(0.45 * M.shape[0] + 2, 0.45 * M.shape[0] + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    if genes is not None:
        ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=7)
    ax.figure.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title or key)
    return ax


def loadings(adata, key="pfa", top=8, ax=None, cmap="coolwarm", title=None):
    """Heatmap of the phylogenetic-factor-analysis loadings ``W`` (``tl.factor_analysis``),
    showing the top-``|loading|`` genes per factor."""
    pfa = adata.uns[key]
    W = np.asarray(pfa["W"]); genes = np.asarray(pfa["genes"])
    sel = np.unique(np.concatenate([np.argsort(-np.abs(W[:, f]))[:top]
                                    for f in range(W.shape[1])]))
    Wt = W[sel]
    if ax is None:
        _, ax = plt.subplots(figsize=(1.4 * W.shape[1] + 2, 0.3 * len(sel) + 1.5))
    lim = np.abs(Wt).max()
    im = ax.imshow(Wt, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(W.shape[1])); ax.set_xticklabels([f"factor {f}" for f in range(W.shape[1])])
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(genes[sel], fontsize=7)
    ax.figure.colorbar(im, ax=ax, shrink=0.7, label="loading")
    ax.set_title(title or "PFA gene loadings")
    return ax
