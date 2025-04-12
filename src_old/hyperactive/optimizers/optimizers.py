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


class HillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _HillClimbingOptimizer


class StochasticHillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _StochasticHillClimbingOptimizer


class RepulsingHillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RepulsingHillClimbingOptimizer


class SimulatedAnnealingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _SimulatedAnnealingOptimizer


class DownhillSimplexOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _DownhillSimplexOptimizer


class RandomSearchOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RandomSearchOptimizer


class GridSearchOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _GridSearchOptimizer


class RandomRestartHillClimbingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RandomRestartHillClimbingOptimizer


class RandomAnnealingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RandomAnnealingOptimizer


class PowellsMethod(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _PowellsMethod


class PatternSearch(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _PatternSearch


class ParallelTemperingOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _ParallelTemperingOptimizer


class ParticleSwarmOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _ParticleSwarmOptimizer


class SpiralOptimization(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _SpiralOptimization_


class GeneticAlgorithmOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _GeneticAlgorithmOptimizer


class EvolutionStrategyOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _EvolutionStrategyOptimizer


class DifferentialEvolutionOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _DifferentialEvolutionOptimizer


class BayesianOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _BayesianOptimizer


class LipschitzOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _LipschitzOptimizer_


class DirectAlgorithm(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _DirectAlgorithm_


class TreeStructuredParzenEstimators(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _TreeStructuredParzenEstimators


class ForestOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _ForestOptimizer


class EnsembleOptimizer(HyperOptimizer):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _EnsembleOptimizer
