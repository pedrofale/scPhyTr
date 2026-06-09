"""Helpers for turning single-cell AnnData into species/clone-level trait tables."""

import numpy as np
import pandas as pd


def _character_per_cell(adata, character):
    """Per-cell values for a character: an obs column or a gene in var_names."""
    if character in adata.obs.columns:
        return np.asarray(adata.obs[character].values, dtype=float)
    if character in adata.var_names:
        col = adata[:, character].X
        col = col.toarray() if hasattr(col, "toarray") else np.asarray(col)
        return np.asarray(col, dtype=float).ravel()
    raise KeyError(f"Character '{character}' not found in adata.obs or adata.var_names.")


def make_trait_table(adata, characters, species_obs="species"):
    """Species x trait table of per-species means (one row per tree leaf).

    Parameters
    ----------
    adata : AnnData
    characters : list[str]
        Gene names (in ``var_names``) or ``obs`` columns to aggregate.
    species_obs : str
        Column in ``adata.obs`` giving the species/clone label of each cell.

    Returns
    -------
    pandas.DataFrame indexed by species, columns = characters.
    """
    species = np.asarray(adata.obs[species_obs].values)
    data = {char: _character_per_cell(adata, char) for char in characters}
    df = pd.DataFrame(data)
    df[species_obs] = species
    return df.groupby(species_obs, observed=True).mean()


def make_trait_values(adata, species_obs, characters):
    """Dict form ``{species: {character: mean}}`` for ``Tree.set_trait_values``."""
    table = make_trait_table(adata, characters, species_obs=species_obs)
    return {sp: row.to_dict() for sp, row in table.iterrows()}
