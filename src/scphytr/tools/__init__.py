# These take anndata objects that contain a tree, meant as API
from .model_selection import (
    FittedModel,
    fit_bm,
    fit_ou,
    fit_ou_regimes,
    fit_bm_counts,
    fit_ou_counts,
    fit_ou_regimes_counts,
    select_model,
    detect_adaptive,
    detect_adaptive_counts,
    fit_bm_rates,
    detect_rate_shifts,
)
from .adaptive import detect_adaptive_genes, detect_adaptive_traits
from .estimation import (
    FittedMVModel,
    fit_bm_mv,
    fit_ou_mv,
    fit_mv,
    fit_mv_latent,
    estimate_rate,
    estimate_correlation,
    estimate_optima,
    cov_to_corr,
)
from .em import fit_mv_em
from .factor_analysis import (
    FittedFactorModel,
    FittedDynamicFactorModel,
    fit_factor_analysis,
    fit_phylo_factor_analysis,
    simulate_pfa,
    detect_factor_dynamics,
    classify_factor_dynamics,
    subspace_error,
    principal_angles,
    procrustes_align,
)
from .poisson_factor import (
    FittedPoissonFactorModel,
    fit_poisson_factor_analysis,
    simulate_poisson_pfa,
)
# AnnData-facing read-outs (scanpy-like tl.*), routed through the modular backend.
# These intentionally shadow the lower-level helpers of the same name (estimate_rate,
# detect_rate_shifts) with the adata-aware versions; detect_rate_shifts dispatches so the
# (tree, values) form still works.
from .api import (
    estimate_rate,
    heritability,
    detect_rate_shifts,
    evolutionary_correlation,
    factor_analysis,
)

__all__ = [
    "FittedModel",
    "fit_bm",
    "fit_ou",
    "fit_ou_regimes",
    "fit_bm_counts",
    "fit_ou_counts",
    "fit_ou_regimes_counts",
    "select_model",
    "detect_adaptive",
    "detect_adaptive_counts",
    "detect_adaptive_genes",
    "detect_adaptive_traits",
    # Multivariate (correlated) BM/OU estimation.
    "FittedMVModel",
    "fit_bm_mv",
    "fit_ou_mv",
    "fit_mv",
    "fit_mv_latent",
    "fit_mv_em",
    "estimate_rate",
    "estimate_correlation",
    "estimate_optima",
    "cov_to_corr",
    # Phylogenetic factor analysis.
    "FittedFactorModel",
    "FittedDynamicFactorModel",
    "fit_factor_analysis",
    "fit_phylo_factor_analysis",
    "simulate_pfa",
    "detect_factor_dynamics",
    "classify_factor_dynamics",
    "subspace_error",
    "principal_angles",
    "procrustes_align",
    # Poisson phylogenetic factor analysis (low-rank latent, raw counts).
    "FittedPoissonFactorModel",
    "fit_poisson_factor_analysis",
    "simulate_poisson_pfa",
    # AnnData-facing read-outs (tl.*).
    "estimate_rate",
    "heritability",
    "detect_rate_shifts",
    "evolutionary_correlation",
    "factor_analysis",
]
