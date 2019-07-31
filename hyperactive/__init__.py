# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

__version__ = "0.4.1.1"
__license__ = "MIT"

from .optimizers.local import HillClimbingOptimizer
from .optimizers.local import StochasticHillClimbingOptimizer
from .optimizers.local import TabuOptimizer

from .optimizers.random import RandomSearchOptimizer
from .optimizers.random import RandomRestartHillClimbingOptimizer
from .optimizers.random import RandomAnnealingOptimizer

from .optimizers.monte_carlo import SimulatedAnnealingOptimizer
from .optimizers.monte_carlo import StochasticTunnelingOptimizer
from .optimizers.monte_carlo import ParallelTemperingOptimizer

from .optimizers.population import ParticleSwarmOptimizer
from .optimizers.population import EvolutionStrategyOptimizer

from .optimizers.sequence_model import BayesianOptimizer

from .chimaera import Chimaera


__all__ = [
    "HillClimbingOptimizer",
    "StochasticHillClimbingOptimizer",
    "TabuOptimizer",
    "RandomSearchOptimizer",
    "RandomRestartHillClimbingOptimizer",
    "RandomAnnealingOptimizer",
    "SimulatedAnnealingOptimizer",
    "StochasticTunnelingOptimizer",
    "ParallelTemperingOptimizer",
    "ParticleSwarmOptimizer",
    "EvolutionStrategyOptimizer",
    "BayesianOptimizer",
    "Chimaera",
]
