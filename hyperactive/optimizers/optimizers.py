# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .gfo_wrapper import _BaseOptimizer_

from gradient_free_optimizers import (
    HillClimbingOptimizer as _HillClimbingOptimizer,
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer as _RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer as _DownhillSimplexOptimizer,
    RandomSearchOptimizer as _RandomSearchOptimizer,
    GridSearchOptimizer as _GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer as _RandomAnnealingOptimizer,
    PowellsMethod as _PowellsMethod,
    PatternSearch as _PatternSearch,
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
    ParticleSwarmOptimizer as _ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
    BayesianOptimizer as _BayesianOptimizer,
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
    ForestOptimizer as _ForestOptimizer,
    EnsembleOptimizer as _EnsembleOptimizer,
)


class HillClimbingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _HillClimbingOptimizer


class StochasticHillClimbingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _StochasticHillClimbingOptimizer


class RepulsingHillClimbingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RepulsingHillClimbingOptimizer


class SimulatedAnnealingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _SimulatedAnnealingOptimizer


class DownhillSimplexOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _DownhillSimplexOptimizer


class RandomSearchOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomSearchOptimizer


class GridSearchOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _GridSearchOptimizer


class RandomRestartHillClimbingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomRestartHillClimbingOptimizer


class RandomAnnealingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomAnnealingOptimizer


class PowellsMethod(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _PowellsMethod


class PatternSearch(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _PatternSearch


class ParallelTemperingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _ParallelTemperingOptimizer


class ParticleSwarmOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _ParticleSwarmOptimizer


class EvolutionStrategyOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _EvolutionStrategyOptimizer


class BayesianOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _BayesianOptimizer


class TreeStructuredParzenEstimators(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _TreeStructuredParzenEstimators


class ForestOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _ForestOptimizer


class EnsembleOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _EnsembleOptimizer
