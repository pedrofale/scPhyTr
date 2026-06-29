"""Plotting (``scphytr.pl``)."""
from .tree import plot_tree, rate_tree
from .heatmaps import matrix, loadings
from .utils import make_species_colors
from .path import trait_paths_1d, trait_paths_2d, trait_paths_tree
from .decomposition import variance_decomposition

__all__ = [
    "plot_tree",
    "rate_tree",
    "matrix",
    "loadings",
    "variance_decomposition",
    "make_species_colors",
    "trait_paths_1d",
    "trait_paths_2d",
    "trait_paths_tree",
]
