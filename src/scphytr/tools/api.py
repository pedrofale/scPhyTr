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


def _leaf_label(adata, obs, leaves, max_states=12):
    """Per-leaf discrete covariate label for ``obs`` (cells -> leaf by majority vote).

    Each leaf (subclone, or a single cell in a single-cell tree) takes the most common value of
    ``adata.obs[obs]`` among its cells. Leaves with no cells fall back to the global majority.
    Raises if the covariate resolves to more than ``max_states`` distinct values (it is meant
    to be categorical -- e.g. clone or niche; bin a continuous covariate first).
    """
    sp = np.asarray(adata.obs[adata.uns["_species_obs"]]).astype(str)
    lab = pd.Series(np.asarray(adata.obs[obs]).astype(str), name="lab")
    df = pd.DataFrame({"sp": sp, "lab": lab})
    maj = df.groupby("sp")["lab"].agg(lambda s: s.value_counts().idxmax())
    fallback = lab.value_counts().idxmax()
    out = {l: str(maj.get(l, fallback)) for l in leaves}
    n_states = len(set(out.values()))
    if n_states > max_states:
        raise ValueError(
            f"covariate '{obs}' resolves to {n_states} states (> max_states={max_states}); "
            "it must be categorical (e.g. clone or niche). Bin a continuous covariate first.")
    return out


def _leaf_trait(adata, character, leaves):
    """Per-leaf trait for ``character`` (a gene -> mean log1p expr, or an obs column -> mean).

    Leaf-level (one value per leaf); used by the Gaussian read-outs (λ, de-novo rate shifts).
    """
    sp = np.asarray(adata.obs[adata.uns["_species_obs"]]).astype(str)
    if character in adata.var_names:
        y = _gene_counts(adata, character)
        s = adata.obs[adata.uns.get("_size_factor_obs", "size_factors")].values
        v = np.log1p(y / np.maximum(np.asarray(s, float), 1e-9))
    elif character in adata.obs:
        v = np.asarray(adata.obs[character].values, dtype=float)
    else:
        raise KeyError(f"character '{character}' is not a gene (var_names) or obs column")
    m = pd.Series(v).groupby(sp).mean()
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


def detect_adaptive(adata, genes=None, regimes=None, n_regimes=None, dispersion=None,
                    models=("BM", "OU", "OU2"), criterion="aic"):
    """Per-gene model selection over the subclonal count model (no pseudobulk).

    Fits the requested ``models`` by the validated count-marginal (cells as replicates) and
    selects by ``criterion``; stores ``var['adaptive_model']`` and ``var['adaptive']`` (OU/OU2
    win). ``"OU2"`` (two-regime adaptive) needs a regime painting (``regimes``, ``n_regimes``
    from ``paint_regimes`` / ``load_regimes``); pass e.g. ``models=("BM","OU2")`` for a direct
    BM-vs-adaptive test.
    """
    tree, leaves, idx, sf = _ctx(adata)
    genes = list(adata.var_names) if genes is None else list(genes)
    crit = (lambda m: m.aic()) if criterion == "aic" else (lambda m: m.bic())
    sel, adaptive = {}, {}
    for g in genes:
        y = _gene_counts(adata, g)
        obs = SubclonalObservation(y, sf, idx, len(leaves), dispersion=dispersion)
        cands = {}
        if "BM" in models:
            cands["BM"] = _ms.fit_bm_counts(tree, obs)
        if "OU" in models:
            cands["OU"] = _ms.fit_ou_counts(tree, obs)
        if "OU2" in models and regimes is not None:
            cands["OU2"] = _ms.fit_ou_regimes_counts(tree, obs, regimes, n_regimes)
        best = min(cands, key=lambda k: crit(cands[k]))
        sel[g], adaptive[g] = best, int(best in ("OU", "OU2"))
    adata.var["adaptive_model"] = pd.Series(sel).reindex(adata.var_names)
    adata.var["adaptive"] = pd.Series(adaptive).reindex(adata.var_names)
    return adata


def plasticity(adata, genes, dispersion=10.0):
    """Heritable vs within-clone plastic variance per gene (the subclonal replicates at work).

    Jointly fits the BM diffusion K and a per-gene within-clone NB dispersion r by Laplace-EM,
    then stores ``var['v_herit']`` (between-clone, K_gg·T), ``var['v_plast']`` (within-clone,
    ψ₁(r)) and ``var['plasticity']`` = V_plast/(V_plast+V_herit) ∈ [0,1].
    """
    from .em import fit_mv_em
    from scipy.special import polygamma
    from ..inference.laplace import MultiCellPoissonObservation
    tree, leaves, idx, sf = _ctx(adata)
    genes = list(genes)
    X = adata[:, genes].X
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(float)
    obs = MultiCellPoissonObservation(X, sf, idx, len(leaves), dispersion=dispersion)
    res = fit_mv_em(tree, obs, model="BM", fit_dispersion=True, max_em=30)
    T = float(tree.root.get_farthest_leaf()[1]) + float(tree.root.dist)
    K = np.asarray(res.covariance()); r = res.extra["dispersion"]
    Vh = np.diag(K) * T; Vp = polygamma(1, r); frac = Vp / (Vp + Vh)
    for col, val in [("v_herit", Vh), ("v_plast", Vp), ("plasticity", frac)]:
        adata.var[col] = pd.Series(dict(zip(genes, val))).reindex(adata.var_names)
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


def covariate_rate_shifts(adata, obs, genes=None, character=None, key="cov_rate"):
    """Test whether a discrete covariate (``adata.obs[obs]``) carries state-specific BM rates.

    The covariate (e.g. ``clone`` or spatial ``niche``) is reconstructed onto the branches by
    parsimony and a state-dependent multi-rate BM is fit and LR-tested against a single global
    rate -- the ML counterpart to RevBayes' state-dependent BM (see
    :func:`scphytr.tools.covariate_rates.fit_covariate_rates`).

    Tests one ``character`` (a gene or obs column) or, by default, every gene in ``genes``
    (all var_names). For a single character the full result is stored in
    ``uns['covariate_rates']`` (with the branch ``regimes`` for ``pl.rate_tree``); for many
    genes, per-gene ``var[key+'_p']`` (LRT p), ``var[key+'_ratio']`` (max/min state rate) and
    ``var[key+'_fastest']`` (fastest state) are written alongside the full results in ``uns``.
    """
    from .covariate_rates import fit_covariate_rates
    tree, leaves, idx, sf = _ctx(adata)
    labels = _leaf_label(adata, obs, leaves)
    single = character is not None
    chars = [character] if single else (list(genes) if genes is not None else list(adata.var_names))
    results, p_, ratio_, fast_ = {}, {}, {}, {}
    for ch in chars:
        vals = _leaf_trait(adata, ch, leaves)
        if np.allclose(list(vals.values()), list(vals.values())[0]):
            continue
        r = fit_covariate_rates(tree, vals, labels)
        results[ch] = r
        p_[ch], ratio_[ch], fast_[ch] = r["p"], r["rate_ratio"], r["fastest_state"]
    store = {"obs": obs, "results": results}
    if single and character in results:
        store.update(results[character])               # regimes/state_names/rates for plotting
        store["character"] = character
    else:
        adata.var[key + "_p"] = pd.Series(p_).reindex(adata.var_names)
        adata.var[key + "_ratio"] = pd.Series(ratio_).reindex(adata.var_names)
        adata.var[key + "_fastest"] = pd.Series(fast_).reindex(adata.var_names)
    adata.uns["covariate_rates"] = store
    return results[character] if single else results


def decompose_variance(adata, genes=None, spatial_key="spatial", dispersion=None,
                       include_residual=False, n_neighbors=6):
    """Partition each gene's expression variance into heritable (tree) vs niche (spatial) vs residual.

    Fits the additive tree⊕space latent-Gaussian model with the subclonal count decoder
    (:func:`scphytr.inference.spatial_decomposition.decompose`) and writes, per gene,
    ``var['v_phylo','v_space','v_resid']`` and ``var['frac_heritable']`` = v_phylo/(v_phylo+v_space).
    Unlike a tree-only rate, this does not misattribute spatial structure to fast evolution. Builds
    the spatial GMRF from ``obsm[spatial_key]`` via :func:`pp.spatial_neighbors` if absent.

    ``include_residual`` is **off by default** on purpose: the iid residual is weakly identifiable
    against the spatial field (both live at the leaves), so enabling it can siphon genuine niche
    variance into the residual and flip a niche gene to ``frac_heritable``≈1. Only turn it on when
    you specifically want to model cell-intrinsic latent variation beyond lineage and niche, and
    treat the resulting split with care.
    """
    from ..inference.spatial_decomposition import decompose
    from ..preprocessing import spatial_neighbors
    tree, leaves, idx, sf = _ctx(adata)
    if "spatial_graph" not in adata.uns:
        spatial_neighbors(adata, spatial_key=spatial_key, n_neighbors=n_neighbors)
    Qs = adata.uns["spatial_graph"]["precision"]
    genes = list(adata.var_names) if genes is None else list(genes)
    vp, vs, vr, fr = {}, {}, {}, {}
    for g in genes:
        y = _gene_counts(adata, g)
        obs = SubclonalObservation(y, sf, idx, len(leaves), dispersion=dispersion)
        d = decompose(tree, obs, Qs, include_residual=include_residual)
        vp[g], vs[g], vr[g], fr[g] = d.v_phylo, d.v_space, d.v_resid, d.frac_heritable
    adata.var["v_phylo"] = pd.Series(vp).reindex(adata.var_names)
    adata.var["v_space"] = pd.Series(vs).reindex(adata.var_names)
    adata.var["v_resid"] = pd.Series(vr).reindex(adata.var_names)
    adata.var["frac_heritable"] = pd.Series(fr).reindex(adata.var_names)
    return adata


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
