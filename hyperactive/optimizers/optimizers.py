# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .hyper_optimizer import HyperOptimizer

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
    SpiralOptimization as _SpiralOptimization_,
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
    BayesianOptimizer as _BayesianOptimizer,
    LipschitzOptimizer as _LipschitzOptimizer_,
    DirectAlgorithm as _DirectAlgorithm_,
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
    ForestOptimizer as _ForestOptimizer,
    EnsembleOptimizer as _EnsembleOptimizer,
)


class HillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _HillClimbingOptimizer


class StochasticHillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _StochasticHillClimbingOptimizer


class RepulsingHillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RepulsingHillClimbingOptimizer


class SimulatedAnnealingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _SimulatedAnnealingOptimizer


class DownhillSimplexOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _DownhillSimplexOptimizer


class RandomSearchOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomSearchOptimizer


class GridSearchOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _GridSearchOptimizer


class RandomRestartHillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomRestartHillClimbingOptimizer


class RandomAnnealingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomAnnealingOptimizer


class PowellsMethod(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _PowellsMethod


class PatternSearch(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _PatternSearch


class ParallelTemperingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _ParallelTemperingOptimizer


class ParticleSwarmOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _ParticleSwarmOptimizer


class SpiralOptimization(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _SpiralOptimization_


class EvolutionStrategyOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _EvolutionStrategyOptimizer


class BayesianOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _BayesianOptimizer


class LipschitzOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _LipschitzOptimizer_


class DirectAlgorithm(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _DirectAlgorithm_


class TreeStructuredParzenEstimators(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _TreeStructuredParzenEstimators


class ForestOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _ForestOptimizer


class EnsembleOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _EnsembleOptimizer
