import matplotlib.pyplot as plt

def make_species_colors(species, colormap='Set1'):
    """
    Make a dictionary of colors for each species.
    """
    return {species: plt.get_cmap(colormap)(i) for i, species in enumerate(species)}