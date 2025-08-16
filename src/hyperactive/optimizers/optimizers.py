"""optimizers module for Hyperactive optimization."""

# Email: simon.blanke@yahoo.com
# License: MIT License


from gradient_free_optimizers import (
    BayesianOptimizer as _BayesianOptimizer,
)
from gradient_free_optimizers import (
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer,
)
from gradient_free_optimizers import (
    DirectAlgorithm as _DirectAlgorithm_,
)
from gradient_free_optimizers import (
    DownhillSimplexOptimizer as _DownhillSimplexOptimizer,
)
from gradient_free_optimizers import (
    EnsembleOptimizer as _EnsembleOptimizer,
)
from gradient_free_optimizers import (
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
)
from gradient_free_optimizers import (
    ForestOptimizer as _ForestOptimizer,
)
from gradient_free_optimizers import (
    GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer,
)
from gradient_free_optimizers import (
    GridSearchOptimizer as _GridSearchOptimizer,
)
from gradient_free_optimizers import (
    HillClimbingOptimizer as _HillClimbingOptimizer,
)
from gradient_free_optimizers import (
    LipschitzOptimizer as _LipschitzOptimizer_,
)
from gradient_free_optimizers import (
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
)
from gradient_free_optimizers import (
    ParticleSwarmOptimizer as _ParticleSwarmOptimizer,
)
from gradient_free_optimizers import (
    PatternSearch as _PatternSearch,
)
from gradient_free_optimizers import (
    PowellsMethod as _PowellsMethod,
)
from gradient_free_optimizers import (
    RandomAnnealingOptimizer as _RandomAnnealingOptimizer,
)
from gradient_free_optimizers import (
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer,
)
from gradient_free_optimizers import (
    RandomSearchOptimizer as _RandomSearchOptimizer,
)
from gradient_free_optimizers import (
    RepulsingHillClimbingOptimizer as _RepulsingHillClimbingOptimizer,
)
from gradient_free_optimizers import (
    SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer,
)
from gradient_free_optimizers import (
    SpiralOptimization as _SpiralOptimization_,
)
from gradient_free_optimizers import (
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer,
)
from gradient_free_optimizers import (
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
)

from .hyper_optimizer import HyperOptimizer


class HillClimbingOptimizer(HyperOptimizer):
    """HillClimbingOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _HillClimbingOptimizer


class StochasticHillClimbingOptimizer(HyperOptimizer):
    """StochasticHillClimbingOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _StochasticHillClimbingOptimizer


class RepulsingHillClimbingOptimizer(HyperOptimizer):
    """RepulsingHillClimbingOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RepulsingHillClimbingOptimizer


class SimulatedAnnealingOptimizer(HyperOptimizer):
    """SimulatedAnnealingOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _SimulatedAnnealingOptimizer


class DownhillSimplexOptimizer(HyperOptimizer):
    """DownhillSimplexOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _DownhillSimplexOptimizer


class RandomSearchOptimizer(HyperOptimizer):
    """RandomSearchOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RandomSearchOptimizer


class GridSearchOptimizer(HyperOptimizer):
    """GridSearchOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _GridSearchOptimizer


class RandomRestartHillClimbingOptimizer(HyperOptimizer):
    """RandomRestartHillClimbingOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RandomRestartHillClimbingOptimizer


class RandomAnnealingOptimizer(HyperOptimizer):
    """RandomAnnealingOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _RandomAnnealingOptimizer


class PowellsMethod(HyperOptimizer):
    """PowellsMethod class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _PowellsMethod


class PatternSearch(HyperOptimizer):
    """PatternSearch class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _PatternSearch


class ParallelTemperingOptimizer(HyperOptimizer):
    """ParallelTemperingOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _ParallelTemperingOptimizer


class ParticleSwarmOptimizer(HyperOptimizer):
    """ParticleSwarmOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _ParticleSwarmOptimizer


class SpiralOptimization(HyperOptimizer):
    """SpiralOptimization class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _SpiralOptimization_


class GeneticAlgorithmOptimizer(HyperOptimizer):
    """GeneticAlgorithmOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _GeneticAlgorithmOptimizer


class EvolutionStrategyOptimizer(HyperOptimizer):
    """EvolutionStrategyOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _EvolutionStrategyOptimizer


class DifferentialEvolutionOptimizer(HyperOptimizer):
    """DifferentialEvolutionOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _DifferentialEvolutionOptimizer


class BayesianOptimizer(HyperOptimizer):
    """BayesianOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _BayesianOptimizer


class LipschitzOptimizer(HyperOptimizer):
    """LipschitzOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _LipschitzOptimizer_


class DirectAlgorithm(HyperOptimizer):
    """DirectAlgorithm class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _DirectAlgorithm_


class TreeStructuredParzenEstimators(HyperOptimizer):
    """TreeStructuredParzenEstimators class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _TreeStructuredParzenEstimators


class ForestOptimizer(HyperOptimizer):
    """ForestOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _ForestOptimizer


class EnsembleOptimizer(HyperOptimizer):
    """EnsembleOptimizer class."""

    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self.optimizer_class = _EnsembleOptimizer
