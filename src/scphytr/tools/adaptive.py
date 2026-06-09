"""AnnData-facing entry points for adaptive-evolution detection.

These build species/clone-level trait tables from an AnnData (whose tree lives in
``adata.uns['tree']``, see ``scphytr.preprocessing.setup_anndata``) and run the
per-trait BM vs OU model selection in ``model_selection.detect_adaptive``.
"""

from .model_selection import detect_adaptive
from .utils import make_trait_table


def _run(adata, characters, species_obs, models, criterion, uns_key):
    tree = adata.uns["tree"]
    trait_table = make_trait_table(adata, characters, species_obs=species_obs)
    trait_table = trait_table.reindex(tree.phylotree.get_leaf_names())
    results = detect_adaptive(tree, trait_table, models=models, criterion=criterion)
    adata.uns[uns_key] = results
    return results


def detect_adaptive_genes(adata, genes, species_obs="species",
                          models=("BM", "OU"), criterion="aic"):
    """For every gene, fit BM and OU-1 and select; flag adaptive genes.

    Stores the result table in ``adata.uns['adaptive_genes']`` and returns it.
    """
    return _run(adata, genes, species_obs, models, criterion, "adaptive_genes")


def detect_adaptive_traits(adata, characters, species_obs="species",
                           models=("BM", "OU"), criterion="aic"):
    """Same as ``detect_adaptive_genes`` but for arbitrary traits (obs columns).

    Stores the result table in ``adata.uns['adaptive_traits']`` and returns it.
    """
    return _run(adata, characters, species_obs, models, criterion, "adaptive_traits")
