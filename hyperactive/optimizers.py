# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd

from gradient_free_optimizers import (
    HillClimbingOptimizer as _HillClimbingOptimizer,
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer as _RepulsingHillClimbingOptimizer,
    RandomSearchOptimizer as _RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer as _RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer,
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
    ParticleSwarmOptimizer as _ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
    BayesianOptimizer as _BayesianOptimizer,
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
    DecisionTreeOptimizer as _DecisionTreeOptimizer,
    EnsembleOptimizer as _EnsembleOptimizer,
)


class DictClass:
    def __init__(self):
        self.para_dict = {}

    def __getitem__(self, key):
        return self.para_dict[key]

    def keys(self):
        return self.para_dict.keys()

    def values(self):
        return self.para_dict.values()


class _BaseOptimizer_(DictClass):
    def __init__(self, **opt_params):
        super().__init__()
        self.opt_params = opt_params

    def init(
        self, search_space, initialize={"grid": 8, "random": 4, "vertices": 8}
    ):
        self.search_space = search_space
        self.optimizer_hyper_ss = self._OptimizerClass(
            search_space, initialize
        )

        search_space_positions = {}
        for key in search_space.keys():
            search_space_positions[key] = np.array(
                range(len(search_space[key]))
            )

        initialize = self._warm_start_conv(initialize)

        self.optimizer = self._OptimizerClass(
            search_space_positions, initialize, **self.opt_params
        )
        self.search_space_positions = search_space_positions

        self.conv = self.optimizer.conv

    def print_info(self, *args):
        self.optimizer.print_info(*args)

    def _warm_start_conv(self, initialize):
        if "warm_start" in list(initialize.keys()):
            warm_start = initialize["warm_start"]
            warm_start_gfo = []
            for warm_start_ in warm_start:
                value = self.optimizer_hyper_ss.conv.para2value(warm_start_)
                position = self.optimizer_hyper_ss.conv.value2position(value)
                pos_para = self.optimizer_hyper_ss.conv.value2para(position)

                warm_start_gfo.append(pos_para)

            initialize["warm_start"] = warm_start_gfo

        return initialize

    def _process_results(self):
        results_dict = {}

        for para_name in self.conv.para_names:
            values_list = self.search_space[para_name]
            pos_ = self.positions[para_name].values
            values_ = [values_list[idx] for idx in pos_]
            results_dict[para_name] = values_

        self.results = pd.DataFrame.from_dict(results_dict)

        diff_list = np.setdiff1d(self.positions.columns, self.results.columns)
        self.results[diff_list] = self.positions[diff_list]

    def search(
        self,
        objective_function,
        n_iter,
        warm_start=None,
        max_time=None,
        max_score=None,
        memory=True,
        memory_warm_start=None,
        verbosity={
            "progress_bar": True,
            "print_results": True,
            "print_times": True,
        },
        random_state=None,
        nth_process=None,
    ):

        self.optimizer.search(
            objective_function,
            n_iter,
            max_time,
            max_score,
            memory,
            memory_warm_start,
            verbosity,
            random_state,
            nth_process,
        )

        self.eval_time = np.array(self.optimizer.eval_times).sum()
        self.iter_time = np.array(self.optimizer.iter_times).sum()

        value = self.optimizer_hyper_ss.conv.para2value(
            self.optimizer.best_para
        )
        position = self.optimizer_hyper_ss.conv.position2value(value)
        best_para = self.optimizer_hyper_ss.conv.value2para(position)

        self.best_para = best_para
        self.best_score = self.optimizer.best_score
        self.positions = self.optimizer.results
        self.memory_dict_new = self.optimizer.memory_dict_new

        self._process_results()


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


class RandomSearchOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomSearchOptimizer


class RandomRestartHillClimbingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomRestartHillClimbingOptimizer


class RandomAnnealingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _RandomAnnealingOptimizer


class SimulatedAnnealingOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _SimulatedAnnealingOptimizer


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


class DecisionTreeOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _DecisionTreeOptimizer


class EnsembleOptimizer(_BaseOptimizer_):
    def __init__(self, **opt_params):
        super().__init__(**opt_params)
        self._OptimizerClass = _EnsembleOptimizer

