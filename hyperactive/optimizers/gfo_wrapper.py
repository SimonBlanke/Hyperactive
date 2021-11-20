# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import copy
import inspect
import numpy as np
import pandas as pd


from .objective_function import ObjectiveFunction
from .hyper_gradient_trafo import HyperGradientTrafo


class TrafoClass:
    def __init__(self, *args, **kwargs):
        pass

    def _convert_args2gfo(self, memory_warm_start):
        memory_warm_start = self.trafo.trafo_memory_warm_start(memory_warm_start)

        return memory_warm_start

    def _positions2results(self, positions):
        results_dict = {}

        for para_name in self.conv.para_names:
            values_list = self.s_space[para_name]
            pos_ = positions[para_name].values
            values_ = [values_list[idx] for idx in pos_]
            results_dict[para_name] = values_

        results = pd.DataFrame.from_dict(results_dict)

        diff_list = np.setdiff1d(positions.columns, results.columns)
        results[diff_list] = positions[diff_list]

        return results

    def _convert_results2hyper(self):
        self.eval_time = np.array(self._optimizer.eval_times).sum()
        self.iter_time = np.array(self._optimizer.iter_times).sum()

        if self._optimizer.best_para is not None:
            value = self.trafo.para2value(self._optimizer.best_para)
            position = self.trafo.position2value(value)
            best_para = self.trafo.value2para(position)

            self.best_para = best_para
        else:
            self.best_para = None

        self.best_score = self._optimizer.best_score
        self.positions = self._optimizer.results

        self.results = self._positions2results(self.positions)

        results_dd = self._optimizer.results.drop_duplicates(
            subset=self.s_space.dim_keys, keep="first"
        )
        self.memory_values_df = results_dd[
            self.s_space.dim_keys + ["score"]
        ].reset_index(drop=True)


class _BaseOptimizer_(TrafoClass):
    def __init__(self, **opt_params):
        super().__init__()
        self.opt_params = opt_params

    def init(self, s_space, initialize, progress_collector):
        self.s_space = s_space
        self.initialize = initialize
        self.progress_collector = progress_collector

        self.trafo = HyperGradientTrafo(s_space)

        initialize = self.trafo.trafo_initialize(initialize)
        search_space_positions = s_space.positions

        # trafo warm start for smbo from values into positions
        if "warm_start_smbo" in self.opt_params:
            self.opt_params["warm_start_smbo"] = self.trafo.trafo_memory_warm_start(
                self.opt_params["warm_start_smbo"]
            )

        self._optimizer = self._OptimizerClass(
            search_space_positions, initialize, **self.opt_params
        )

        self.conv = self._optimizer.conv

    def search(
        self,
        objective_function,
        n_iter,
        max_time=None,
        max_score=None,
        early_stopping=None,
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
        self.objective_function = objective_function
        self.nth_process = nth_process

        gfo_wrapper_model = ObjectiveFunction(
            objective_function, self._optimizer, nth_process
        )

        memory_warm_start = self._convert_args2gfo(memory_warm_start)

        gfo_objective_function = gfo_wrapper_model(
            self.s_space(), self.progress_collector
        )

        self._optimizer.search(
            objective_function=gfo_objective_function,
            n_iter=n_iter,
            max_time=max_time,
            max_score=max_score,
            early_stopping=early_stopping,
            memory=memory,
            memory_warm_start=memory_warm_start,
            verbosity=verbosity,
            random_state=random_state,
            nth_process=nth_process,
        )

        self._convert_results2hyper()
        self.p_bar = self._optimizer.p_bar
