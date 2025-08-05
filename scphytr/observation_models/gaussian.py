import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class Gaussian(object):
    def __init__(self, observations, trait_values, learnable_parameters=['stds']):
        self.observations = observations
        self.trait_values = trait_values
        self.learnable_parameters = learnable_parameters

    def simulate_observations(self, seed=42):
        np.random.seed(seed)
        return np.random.normal(self.trait_values, self.stds)

    def score(self, stds):
        return multivariate_normal.logpdf(self.observations, self.trait_values, stds)
