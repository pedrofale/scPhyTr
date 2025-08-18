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

__version__ = "0.0.0"
__all__ = [
    "utils",
    "inference", 
    "observation_models",
    "plotting",
    "preprocessing",
    "tools",
    "trait_models",
    "Tree",
]
