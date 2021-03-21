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

from .hyper_gradient_trafo import HyperGradientTrafo


def gfo2hyper(search_space, para):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class DictClass:
    def __init__(self):
        self.para_dict = {}

    def __getitem__(self, key):
        return self.para_dict[key]

    def keys(self):
        return self.para_dict.keys()

    def values(self):
        return self.para_dict.values()


class TrafoClass:
    def __init__(self, *args, **kwargs):
        pass

    def _convert_args2gfo(self, memory_warm_start):
        memory_warm_start = self.trafo.trafo_memory_warm_start(memory_warm_start)

        return memory_warm_start

    def _positions2results(self, positions):
        results_dict = {}

        for para_name in self.conv.para_names:
            values_list = self.search_space[para_name]
            pos_ = positions[para_name].values
            values_ = [values_list[idx] for idx in pos_]
            results_dict[para_name] = values_

        results = pd.DataFrame.from_dict(results_dict)

        diff_list = np.setdiff1d(positions.columns, results.columns)
        results[diff_list] = positions[diff_list]

        return results

    def _convert_results2hyper(self):
        self.eval_time = np.array(self.optimizer.eval_times).sum()
        self.iter_time = np.array(self.optimizer.iter_times).sum()

        if self.optimizer.best_para is not None:
            value = self.trafo.para2value(self.optimizer.best_para)
            position = self.trafo.position2value(value)
            best_para = self.trafo.value2para(position)

            self.best_para = best_para
        else:
            self.best_para = None

        self.best_score = self.optimizer.best_score
        self.positions = self.optimizer.results

        self.results = self._positions2results(self.positions)

        results_dd = self.optimizer.results.drop_duplicates(
            subset=self.trafo.para_names, keep="first"
        )
        self.memory_values_df = results_dd[
            self.trafo.para_names + ["score"]
        ].reset_index(drop=True)


class _BaseOptimizer_(DictClass, TrafoClass):
    def __init__(self, **opt_params):
        super().__init__()
        self.opt_params = opt_params

    def init(self, search_space, initialize={"grid": 8, "random": 4, "vertices": 8}):
        self.search_space = search_space

        self.trafo = HyperGradientTrafo(search_space)

        initialize = self.trafo.trafo_initialize(initialize)
        search_space_positions = self.trafo.search_space_positions

        # trafo warm start for smbo from values into positions
        if "warm_start_smbo" in self.opt_params:
            self.opt_params["warm_start_smbo"] = self.trafo.trafo_memory_warm_start(
                self.opt_params["warm_start_smbo"]
            )

        self.optimizer = self._OptimizerClass(
            search_space_positions, initialize, **self.opt_params
        )

        self.conv = self.optimizer.conv

    def print_info(self, *args):
        self.optimizer.print_info(*args)

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

        """
        import copy
        import inspect

        if inspect.isclass(type(memory)):
            print(memory, type(memory))
            print("Long Term Memory")
            ltm = copy.deepcopy(memory)
            ltm._get_data_types(self.search_space)

            memory_warm_start = ltm._load()

            ltm._init_data_path(objective_function, nth_process)
            memory = True

            if ltm.save_on == "iteration":

                def ltm_wrapper(results, para):
                    if isinstance(results, tuple):
                        score = results[0]
                        results_dict = results[1]
                    else:
                        score = results
                        results_dict = {}

                    results_dict["score"] = score
                    ltm_dict = {**para, **results_dict}
                    ltm._append(ltm_dict)

            else:

                def ltm_wrapper(results, para):
                    pass

        print("\n self.search_space \n", self.search_space, "\n")

        print("\n memory_warm_start \n", memory_warm_start, "\n")
        """

        memory_warm_start = self._convert_args2gfo(memory_warm_start)

        def gfo_wrapper_model():
            # wrapper for GFOs
            def _model(para):
                para = gfo2hyper(self.search_space, para)
                self.para_dict = para
                results = objective_function(self)

                # ltm_wrapper(results, para)

                return results

            _model.__name__ = objective_function.__name__
            return _model

        self.optimizer.search(
            gfo_wrapper_model(),
            n_iter,
            max_time,
            max_score,
            memory,
            memory_warm_start,
            verbosity,
            random_state,
            nth_process,
        )

        self._convert_results2hyper()

        """
        if inspect.isclass(type(memory)):
            ltm._save(self.results)
        """


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
