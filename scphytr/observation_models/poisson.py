import numpy as np
import pandas as pd
from scipy.stats import poisson

class Poisson(object):
    def __init__(self, observations, trait_values, learnable_parameters=['cell_scales', 'gene_scales']):
        self.observations = observations
        self.trait_values = trait_values
        self.learnable_parameters = learnable_parameters

    def simulate_observations(self, seed=42):
        np.random.seed(seed)
        return np.random.poisson(self.cell_scales * self.gene_scales * np.exp(self.trait_values))

    def score(self, cell_scales, gene_scales):
        return poisson.logpdf(self.observations, cell_scales * gene_scales * np.exp(self.trait_values))
