from .poisson import Poisson
from .gaussian import Gaussian
from .subclonal import SubclonalObservation, NegativeBinomial

observation_models = {
    'poisson': Poisson,
    'gaussian': Gaussian,
    'subclonal': SubclonalObservation,
    'negative_binomial': NegativeBinomial,
}