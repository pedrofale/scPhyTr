import matplotlib.pyplot as plt
import numpy as np
from .utils import make_species_colors

def trait_paths_1d(adata, traits=None, draw=True, **kwargs):
    # Grab the path from adata.uns['paths']
    paths = adata.uns['paths']

    # Get species colors
    if 'species_colors' in adata.uns:
        species_colors = {species: adata.uns['species_colors'][species] for species in adata.obs['species'].unique()}
    else:
        species_colors = make_species_colors(adata.obs['species'].unique())

    # Get species order from phylogenetic tree
    species_order = adata.uns['tree'].get_leaf_names()

    # Plot the paths
    return plot_1d(paths, traits=traits, species_order=species_order, species_colors=species_colors, draw=draw, **kwargs)

def trait_paths_2d(adata, traits=[0, 1], draw=True, **kwargs):
    # Grab the path from adata.uns['paths']
    paths = adata.uns['paths']

    # Get species colors
    if 'species_colors' in adata.uns:
        species_colors = {species: adata.uns['species_colors'][species] for species in adata.obs['species'].unique()}
    else:
        species_colors = make_species_colors(adata.obs['species'].unique())

    # Get species order from phylogenetic tree
    species_order = adata.uns['tree'].get_leaf_names()

    # Plot the paths
    return plot_2d(paths, traits=traits, species_order=species_order, species_colors=species_colors, draw=draw, **kwargs)

def trait_paths_tree(adata, draw=True, **kwargs):
    pass

def plot_1d(paths, traits=None, species_order=None, species_colors=None, draw=True, **kwargs):    
    """
    Plot trait paths for each species.
    Parameters
    ----------
    paths : dict
        Dictionary of trait paths for each species. Keys are species names, values are numpy arrays of trait paths.
    traits : list, optional
        List of traits to plot. Default is None, which plots all traits.
    species_colors : dict, optional
        Dictionary of colors for each species. Keys are species names, values are colors.
    draw : bool, optional
        Whether to draw the plot. Default is True.
    **kwargs : dict, optional
        Keyword arguments for plt.subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : matplotlib.axes.Axes
        The axes object.
    """
    if traits is None:
        traits = list(range(paths[next(iter(paths))].shape[1]))
    else:
        paths = {species: paths[species][:, traits] for species in paths}
    n_traits = len(traits)
    
    fig, axes = plt.subplots(n_traits, 1, sharex=True) # One trait per row
    for i in range(n_traits):
        if n_traits > 1:
            plt.sca(axes[i])
        for species in species_order:
            plt.plot(paths[species][:,traits[i]], label=f'{species}', c=species_colors[species])
        plt.title(f'Trait {traits[i]}' if traits is not None else f'Trait {i}')
    plt.xlabel('Time')
    plt.legend(title='Species')
    if draw:
        plt.show()
    return fig, axes

def plot_2d(paths, traits=[0, 1], species_order=None, cmap='viridis', draw=True, **kwargs):    
    """
    Plot trait paths for each species.
    Parameters
    ----------
    paths : dict
        Dictionary of trait paths for each species. Keys are species names, values are numpy arrays of trait paths.
    traits : list, optional
        List of traits to plot. Default is [0, 1], which plots the first two traits.
    species_order : list, optional
        List of species names in the order they should be plotted.
    cmap : str, optional
        Colormap to use for the scatter plot.
    draw : bool, optional
        Whether to draw the plot. Default is True.
    **kwargs : dict, optional
        Keyword arguments for plt.subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : matplotlib.axes.Axes
        The axes object.
    """
    # View pairwise trait paths
    fig, axes = plt.subplots(1, len(paths), figsize=(12, 3), sharex=True, sharey=True)
    for i, species in enumerate(species_order):
        plt.sca(axes[i])
        plt.scatter(paths[species][:,traits[0]], paths[species][:,traits[1]], label=species, c=np.linspace(0, 1, len(paths[species][:,0])), cmap=cmap)
        plt.xlabel('Trait 1')
        plt.ylabel('Trait 2')
        plt.title(species)
    plt.colorbar(label='Time')
    plt.suptitle('Pairwise trait paths')
    if draw:
        plt.show()
    return fig, axes