"""AnnData-facing ``scphytr.tl`` read-outs, routed through the modular backend.

Each function builds a *trait model* × *observation model* and runs an *inference algorithm*
(default the fast :class:`scphytr.inference.Laplace`) on the AnnData, storing results in
``var``/``uns``. The count read-outs use the **subclonal** observation (cells as replicates of
their leaf), never collapsing to pseudobulk. ``pp.setup_anndata`` must have been called.
"""
import numpy as np
import pandas as pd

from ..inference import Laplace
from ..observation_models import SubclonalObservation
from ..trait_models import BrownianMotion
from . import model_selection as _ms
from . import heritability as _herit


def _ctx(adata):
    tree = adata.uns["tree"]
    leaves = tree.phylotree.get_leaf_names()
    idx = np.asarray(adata.obs["_leaf_index"].values, dtype=int)
    sf = np.asarray(adata.obs[adata.uns.get("_size_factor_obs", "size_factors")].values, float)
    return tree, leaves, idx, sf


def _gene_counts(adata, gene):
    X = adata[:, gene].X
    return (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(float).ravel()


def _leaf_trait(adata, gene, leaves):
    """Per-leaf mean log1p expression (leaf-level; used by the Gaussian read-outs)."""
    y = _gene_counts(adata, gene)
    s = adata.obs[adata.uns.get("_size_factor_obs", "size_factors")].values
    ln = np.log1p(y / np.maximum(s, 1e-9))
    df = pd.DataFrame({"v": ln, "s": np.asarray(adata.obs[adata.uns["_species_obs"]]).astype(str)})
    m = df.groupby("s")["v"].mean()
    return {l: float(m.get(l, 0.0)) for l in leaves}


# --------------------------------------------------------------------------- #
# Read-outs
# --------------------------------------------------------------------------- #

def estimate_rate(adata, genes=None, dispersion=None, key="rate"):
    """Per-gene BM diffusion rate σ² via modular Laplace on the subclonal count model.

    Stores ``adata.var[key]``. ``dispersion=r`` uses the NB (within-clone) observation.
    """
    tree, leaves, idx, sf = _ctx(adata)
    genes = list(adata.var_names) if genes is None else list(genes)
    out = {}
    for g in genes:
        y = _gene_counts(adata, g)
        obs = SubclonalObservation(y, sf, idx, len(leaves), dispersion=dispersion)
        bm = BrownianMotion(tree, np.array([float(np.log1p(y.mean() + 1e-9))]),
                            np.array([[1.0]]), learnable_parameters=["rates", "means"])
        Laplace(tree, bm, obs).fit()
        out[g] = bm.process_params()["sigma2"]
    adata.var[key] = pd.Series(out).reindex(adata.var_names)
    return adata


def heritability(adata, genes=None):
    """Per-gene Pagel's λ (+ LR p-value) -> ``adata.var['lambda','lambda_p']``.

    NOTE: λ is a *leaf-level* statistic (one value per leaf), so it reads the per-leaf mean
    expression; the replicate-aware heritability is the count-model V_herit (see plasticity).
    """
    tree, leaves, idx, sf = _ctx(adata)
    C = _herit.shared_ancestry_cov(tree)
    genes = list(adata.var_names) if genes is None else list(genes)
    lam, pval = {}, {}
    for g in genes:
        vals = _leaf_trait(adata, g, leaves)
        if np.allclose(list(vals.values()), list(vals.values())[0]):
            continue
        r = _herit.pagels_lambda(tree, vals, C=C)
        lam[g], pval[g] = r["lambda"], r["p"]
    adata.var["lambda"] = pd.Series(lam).reindex(adata.var_names)
    adata.var["lambda_p"] = pd.Series(pval).reindex(adata.var_names)
    return adata


def detect_rate_shifts(data, *args, character=None, max_shifts=4, criterion="bic", **kw):
    """De-novo clade rate-shift detection. Dispatches on the first argument:

    * ``detect_rate_shifts(adata, character=gene_or_obs)`` -> stores ``uns['rate_shifts']``
      (and is read by ``pl.rate_tree(adata)``).
    * ``detect_rate_shifts(tree, values, ...)`` -> the standalone fit (returns the result dict).
    """
    if not hasattr(data, "uns"):                      # (tree, values) standalone form
        return _ms.detect_rate_shifts(data, *args, max_shifts=max_shifts,
                                      criterion=criterion, **kw)
    adata = data
    tree, leaves, idx, sf = _ctx(adata)
    if character is None:
        raise ValueError("pass character= (a gene in var_names or an obs column)")
    vals = _leaf_trait(adata, character, leaves)
    res = _ms.detect_rate_shifts(tree, vals, max_shifts=max_shifts, criterion=criterion, **kw)
    adata.uns["rate_shifts"] = res
    adata.uns["rate_shifts_character"] = character
    return res


def evolutionary_correlation(adata, genes, dispersion=None, key="K"):
    """Deconfounded gene-gene evolutionary correlation K via the multivariate count model.

    Uses ``fit_mv_em`` (multivariate Laplace-EM, subclonal obs); stores ``uns[key]`` (covariance)
    and ``uns[key+'_corr']`` (correlation) with ``uns[key+'_genes']``.
    """
    from .em import fit_mv_em
    from ..inference.laplace import MultiCellPoissonObservation
    tree, leaves, idx, sf = _ctx(adata)
    genes = list(genes)
    X = adata[:, genes].X
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(float)
    obs = MultiCellPoissonObservation(X, sf, idx, len(leaves), dispersion=dispersion)
    res = fit_mv_em(tree, obs, model="BM", max_em=30)
    K = np.asarray(res.covariance())
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    adata.uns[key] = K
    adata.uns[key + "_corr"] = K / np.outer(d, d)
    adata.uns[key + "_genes"] = list(genes)
    return adata


def factor_analysis(adata, k=5, genes=None, key="pfa"):
    """Poisson phylogenetic factor analysis (low-rank K = W Wᵀ) -> ``uns[key]``."""
    from .poisson_factor import fit_poisson_factor_analysis
    tree, leaves, idx, sf = _ctx(adata)
    genes = list(adata.var_names) if genes is None else list(genes)
    # subclone-summed counts (sufficient statistic) for the factor fit
    X = adata[:, genes].X
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(float)
    Y = np.zeros((len(leaves), len(genes)))
    for c in range(X.shape[0]):
        Y[idx[c]] += X[c]
    fm = fit_poisson_factor_analysis(Y, tree, k, leaf_names=leaves)
    adata.uns[key] = {"W": fm.W, "scores": fm.scores, "genes": list(genes),
                      "K_corr": fm.evolutionary_correlation()}
    return adata
