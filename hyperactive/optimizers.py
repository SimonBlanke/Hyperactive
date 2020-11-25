# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import (
    HillClimbingOptimizer as _HillClimbingOptimizer,
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer,
    TabuOptimizer as _TabuOptimizer,
    RandomSearchOptimizer as _RandomSearchOptimizer,
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


class _BaseOptimizer_:
    def __init__(self, **opt_params):
        self.opt_params = opt_params

    def init(self, search_space):
        search_space_positions = {}
        for key in search_space.keys():
            search_space_positions[key] = np.array(
                range(len(search_space[key]))
            )

        self.optimizer = self._OptimizerClass(
            search_space_positions, **self.opt_params
        )

    def search(self, *args, **kwargs):
        self.optimizer.search(*args, **kwargs)

        self.best_para = self.optimizer.best_para
        self.best_score = self.optimizer.best_score
        self.results = self.optimizer.results
        self.memory_dict_new = self.optimizer.memory_dict_new


class HillClimbingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _HillClimbingOptimizer


class StochasticHillClimbingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _StochasticHillClimbingOptimizer


class TabuOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _TabuOptimizer


class RandomSearchOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomSearchOptimizer

