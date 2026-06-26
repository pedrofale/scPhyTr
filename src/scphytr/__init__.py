# Import key modules to make them accessible at package level
from . import utils
from . import inference
from . import observation_models
from . import plotting
from . import preprocessing
from . import tools
from . import trait_models

# Make commonly used classes available directly
from .utils.tree import Tree
from .inference import (
    LaplaceEStep, ImportanceSamplingEStep, MCMCEStep, resolve_estep,
)

# Scanpy-style short aliases (see README usage examples).
tl = tools
pp = preprocessing
pl = plotting

from .simulation import simulate, simulate_anndata

__version__ = "0.0.0"
__all__ = [
    "simulate",
    "simulate_anndata",
    "utils",
    "inference",
    "observation_models",
    "plotting",
    "preprocessing",
    "tools",
    "trait_models",
    "Tree",
    "tl",
    "pp",
    "pl",
    "LaplaceEStep",
    "ImportanceSamplingEStep",
    "MCMCEStep",
    "resolve_estep",
]
