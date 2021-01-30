# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

__version__ = "3.1.0a1"
__license__ = "MIT"


from .hyperactive import Hyperactive
from .optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    DecisionTreeOptimizer,
    EnsembleOptimizer,
)

# from .long_term_memory import LongTermMemory, Dashboard


__all__ = [
    "Hyperactive",
    # "LongTermMemory",
    # "Dashboard",
    "HillClimbingOptimizer",
    "StochasticHillClimbingOptimizer",
    "RepulsingHillClimbingOptimizer",
    "RandomSearchOptimizer",
    "RandomRestartHillClimbingOptimizer",
    "RandomAnnealingOptimizer",
    "SimulatedAnnealingOptimizer",
    "ParallelTemperingOptimizer",
    "ParticleSwarmOptimizer",
    "EvolutionStrategyOptimizer",
    "BayesianOptimizer",
    "TreeStructuredParzenEstimators",
    "DecisionTreeOptimizer",
    "EnsembleOptimizer",
]
