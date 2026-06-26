from .brownian_motion import BrownianMotion
from .ornstein_uhlenbeck import OrnsteinUhlenbeck
from .multi_rate_brownian_motion import MultiRateBM

trait_models = {
    'brownian_motion': BrownianMotion,
    'ornstein_uhlenbeck': OrnsteinUhlenbeck,
    'multi_rate_brownian_motion': MultiRateBM,
}