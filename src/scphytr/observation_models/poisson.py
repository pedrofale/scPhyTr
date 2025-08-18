import numpy as np
import pandas as pd
from scipy.stats import poisson

from .base import BaseObservationModel

class Poisson(BaseObservationModel):
    def __init__(self, observations, trait_values, learnable_parameters=['cell_scales', 'gene_scales']):
        self.observations = observations
        self.trait_values = trait_values
        self.learnable_parameters = learnable_parameters

    def simulate_observations(self, seed=42):
        np.random.seed(seed)
        return np.random.poisson(self.cell_scales * self.gene_scales * np.exp(self.trait_values))

    def score(self, cell_scales, gene_scales):
        return poisson.logpdf(self.observations, cell_scales * gene_scales * np.exp(self.trait_values))

    def sample_parameters(self):
        return self.cell_scales, self.gene_scales

    def set_parameters(self, params):
        self.cell_scales, self.gene_scales = params

    def kl_divergence(self, params):
        return 0

    def log_likelihood(self, x, params, observation_sample, trait_sample):
        return poisson.logpdf(x, params[0] * params[1] * np.exp(trait_sample))