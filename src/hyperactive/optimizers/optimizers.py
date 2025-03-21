# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._optimizer_api import BaseOptimizer

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
    GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer,
    BayesianOptimizer as _BayesianOptimizer,
    LipschitzOptimizer as _LipschitzOptimizer_,
    DirectAlgorithm as _DirectAlgorithm_,
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
    ForestOptimizer as _ForestOptimizer,
    EnsembleOptimizer as _EnsembleOptimizer,
)


class HillClimbingOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_HillClimbingOptimizer, opt_params)


class StochasticHillClimbingOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_StochasticHillClimbingOptimizer, opt_params)


class RepulsingHillClimbingOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_RepulsingHillClimbingOptimizer, opt_params)


class SimulatedAnnealingOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_SimulatedAnnealingOptimizer, opt_params)


class DownhillSimplexOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_DownhillSimplexOptimizer, opt_params)


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_RandomSearchOptimizer, opt_params)


class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_GridSearchOptimizer, opt_params)


class RandomRestartHillClimbingOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_RandomRestartHillClimbingOptimizer, opt_params)


class RandomAnnealingOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_RandomAnnealingOptimizer, opt_params)


class PowellsMethod(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_PowellsMethod, opt_params)


class PatternSearch(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_PatternSearch, opt_params)


class ParallelTemperingOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_ParallelTemperingOptimizer, opt_params)


class ParticleSwarmOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_ParticleSwarmOptimizer, opt_params)


class SpiralOptimization(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_SpiralOptimization_, opt_params)


class GeneticAlgorithmOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_GeneticAlgorithmOptimizer, opt_params)


class EvolutionStrategyOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_EvolutionStrategyOptimizer, opt_params)


class DifferentialEvolutionOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_DifferentialEvolutionOptimizer, opt_params)


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_BayesianOptimizer, opt_params)


class LipschitzOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_LipschitzOptimizer_, opt_params)


class DirectAlgorithm(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_DirectAlgorithm_, opt_params)


class TreeStructuredParzenEstimators(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_TreeStructuredParzenEstimators, opt_params)


class ForestOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_ForestOptimizer, opt_params)


class EnsembleOptimizer(BaseOptimizer):
    def __init__(self, **opt_params):
        super().__init__(_EnsembleOptimizer, opt_params)
