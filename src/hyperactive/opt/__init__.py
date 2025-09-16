"""Individual optimization algorithms."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.opt.gridsearch import GridSearchSk
from hyperactive.opt.random_search import RandomSearchSk

from .gfo import (
    BayesianOptimizer,
    DifferentialEvolution,
    DirectAlgorithm,
    DownhillSimplexOptimizer,
    EvolutionStrategy,
    ForestOptimizer,
    GeneticAlgorithm,
    GridSearch,
    HillClimbing,
    LipschitzOptimizer,
    ParallelTempering,
    ParticleSwarmOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomRestartHillClimbing,
    RandomSearch,
    RepulsingHillClimbing,
    SimulatedAnnealing,
    SpiralOptimization,
    StochasticHillClimbing,
    TreeStructuredParzenEstimators,
)

__all__ = [
    "GridSearchSk",
    "RandomSearchSk",
    "HillClimbing",
    "RepulsingHillClimbing",
    "StochasticHillClimbing",
    "SimulatedAnnealing",
    "DownhillSimplexOptimizer",
    "RandomSearch",
    "GridSearch",
    "RandomRestartHillClimbing",
    "PowellsMethod",
    "PatternSearch",
    "LipschitzOptimizer",
    "DirectAlgorithm",
    "ParallelTempering",
    "ParticleSwarmOptimizer",
    "SpiralOptimization",
    "GeneticAlgorithm",
    "EvolutionStrategy",
    "DifferentialEvolution",
    "BayesianOptimizer",
    "TreeStructuredParzenEstimators",
    "ForestOptimizer",
]

try:
    from .optuna import (
        TPEOptimizer,
        RandomOptimizer,
        CmaEsOptimizer,
        GPOptimizer,
        GridOptimizer,
        NSGAIIOptimizer,
        NSGAIIIOptimizer,
        QMCOptimizer,
    )

    __all__ += [
        "TPEOptimizer",
        "RandomOptimizer",
        "CmaEsOptimizer",
        "GPOptimizer",
        "GridOptimizer",
        "NSGAIIOptimizer",
        "NSGAIIIOptimizer",
        "QMCOptimizer",
    ]
except ImportError:  # optuna not available; skip re-exports
    pass
