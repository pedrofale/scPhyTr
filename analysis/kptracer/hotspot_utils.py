"""Helpers to run Hotspot (DeTomaso & Yosef 2021) and compare its gene-gene
association matrix to a phylogeny-deconfounded estimate.

The point of these utilities is a single, apples-to-apples object: every method
produces a gene x gene matrix of association *Z-scores*, which we turn into
two-sided p-values and Benjamini-Hochberg-correct identically. The methods then
differ only in what they regress against:

  * ``hotspot``  -- graph-smoothed tip cross-product Sum_ij W_ij x_i y_j, with W a
    cell-cell adjacency (an explicit phylogeny in tree mode, or a KNN graph in a
    latent space). Hotspot's null is *cell exchangeability*: it assumes a gene's
    values are i.i.d. across cells. On a phylogeny that is false, so the null
    variance is under-estimated and Z is over-dispersed (manufactured modules).
  * ``tip``      -- the naive Pearson correlation at the leaves, the thing one
    gets by correlating two genes' expression across cells. Also confounded:
    it estimates a quantity proportional to the tip covariance C (X) K, not K.
  * ``contrast`` -- Felsenstein's independent contrasts: whiten the leaves by the
    phylogenetic covariance C, leaving n-1 i.i.d. rows whose correlation is the
    deconfounded evolutionary correlation (the off-diagonal of K). This is the
    target scPhyTr estimates directly.

Under evolutionarily independent genes (K diagonal, no true module), ``contrast``
is calibrated while ``hotspot`` and ``tip`` over-call.
"""

import numpy as np
import anndata
from scipy import stats


def build_adata(Y, names, genes=None):
    """AnnData (cells x genes) with obs_names = tree leaf labels."""
    Y = np.asarray(Y, dtype="float64")
    ad = anndata.AnnData(X=Y.copy())
    ad.obs_names = list(map(str, names))
    ad.var_names = list(map(str, genes)) if genes is not None else [f"g{j}" for j in range(Y.shape[1])]
    return ad


def run_hotspot(Y, names, *, gene_names=None, model="normal", tree=None, latent=None,
                restrict_genes=None, n_neighbors=None, jobs=1, fdr=0.05):
    """Run Hotspot in tree mode (pass ``tree``) or KNN mode (pass ``latent``).

    ``gene_names`` labels the columns of ``Y`` (so Hotspot's module/index labels
    match the caller's gene names). The pairwise local-correlation step runs on
    ``restrict_genes`` if given, else on genes with autocorrelation FDR < ``fdr``.

    Returns dict with ``autocorr`` (DataFrame), ``lcz`` (gene x gene local
    correlation Z, DataFrame), the fitted ``hs`` object, and ``sig_genes``.
    """
    import hotspot as hotspot_mod

    ad = build_adata(Y, names, genes=gene_names)
    n = ad.shape[0]
    if n_neighbors is None:
        n_neighbors = max(5, int(np.sqrt(n)))

    if tree is not None:
        hs = hotspot_mod.Hotspot(ad, model=model, tree=tree)
    elif latent is not None:
        ad.obsm["X_lat"] = np.asarray(latent, dtype="float64")
        hs = hotspot_mod.Hotspot(ad, model=model, latent_obsm_key="X_lat")
    else:
        raise ValueError("Provide either tree= or latent=.")

    hs.create_knn_graph(weighted_graph=False, n_neighbors=n_neighbors)
    ac = hs.compute_autocorrelations(jobs=jobs)
    sig = list(restrict_genes) if restrict_genes is not None else list(ac.index[ac.FDR < fdr])
    lcz = None
    if len(sig) >= 2:
        hs.compute_local_correlations(sig, jobs=jobs)
        lcz = hs.local_correlation_z.copy()
    return {"autocorr": ac, "lcz": lcz, "hs": hs, "sig_genes": sig}


# --------------------------------------------------------------------------- #
# Phylogeny-aware and naive correlation estimators (gene x gene)
# --------------------------------------------------------------------------- #

def _orthonormal_contrasts(Y, C):
    """Felsenstein's n-1 standardized contrasts: i.i.d. rows under independent BM.

    H is an orthonormal basis of the mean-zero subspace (H^T 1 = 0); whitening by
    chol(H^T C H) removes both the unknown root mean and the tree covariance.
    """
    n = C.shape[0]
    M = np.eye(n) - np.ones((n, n)) / n
    Uc, _, _ = np.linalg.svd(M)
    H = Uc[:, : n - 1]
    G = np.linalg.cholesky(H.T @ C @ H + 1e-10 * np.eye(n - 1))
    return np.linalg.solve(G, H.T @ Y)            # (n-1, p), i.i.d. rows


def contrast_corr(Y, C):
    """Deconfounded gene-gene correlation + effective sample size (m = n-1)."""
    Z = _orthonormal_contrasts(Y, C)
    return np.corrcoef(Z, rowvar=False), Z.shape[0]


def tip_corr(Y):
    """Naive Pearson correlation across cells at the leaves (m = n)."""
    return np.corrcoef(Y, rowvar=False), Y.shape[0]


def corr_to_z(R, m):
    """Fisher z-transform: atanh(r) * sqrt(m-3) ~ N(0,1) under r=0, i.i.d. rows."""
    R = np.clip(R, -0.999999, 0.999999)
    return np.arctanh(R) * np.sqrt(max(m - 3, 1))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (1-D)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def upper_tri_z(Zmat, genes=None):
    """Off-diagonal (upper-triangle) Z-scores of a gene x gene matrix.

    Accepts a numpy array or a pandas DataFrame (Hotspot's local_correlation_z).
    Returns (zvals, (i_idx, j_idx)) so callers can map pairs back to gene names.
    """
    Z = np.asarray(Zmat.values if hasattr(Zmat, "values") else Zmat, float)
    iu = np.triu_indices_from(Z, 1)
    return Z[iu], iu


def n_significant_pairs(zvals, alpha=0.05):
    """# gene pairs significant after BH-FDR on two-sided p from |z|."""
    if zvals is None or len(zvals) == 0:
        return 0, np.array([])
    p = 2 * stats.norm.sf(np.abs(zvals))
    q = bh_fdr(p)
    return int((q < alpha).sum()), q
