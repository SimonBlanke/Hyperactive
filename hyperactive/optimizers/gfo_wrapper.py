# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import copy
import inspect
import numpy as np
import pandas as pd


from .objective_function import ObjectiveFunction
from .hyper_gradient_conv import HyperGradientConv


class TrafoClass:
    def __init__(self, *args, **kwargs):
        pass

    def _convert_args2gfo(self, memory_warm_start):
        memory_warm_start = self.hg_conv.conv_memory_warm_start(memory_warm_start)

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
        self.eval_times = np.array(self._optimizer.eval_times).sum()
        self.iter_times = np.array(self._optimizer.iter_times).sum()

        if self._optimizer.best_para is not None:
            value = self.hg_conv.para2value(self._optimizer.best_para)
            position = self.hg_conv.position2value(value)
            best_para = self.hg_conv.value2para(position)

            self.best_para = best_para
        else:
            self.best_para = None

        self.best_score = self._optimizer.best_score
        self.positions = self._optimizer.search_data

        self.search_data = self._positions2results(self.positions)

        results_dd = self._optimizer.search_data.drop_duplicates(
            subset=self.s_space.dim_keys, keep="first"
        )
        self.memory_values_df = results_dd[
            self.s_space.dim_keys + ["score"]
        ].reset_index(drop=True)


class _BaseOptimizer_(TrafoClass):
    def __init__(self, **opt_params):
        super().__init__()
        self.opt_params = opt_params

    def setup_search(
        self,
        objective_function,
        s_space,
        n_iter,
        initialize,
        pass_through,
        max_score,
        early_stopping,
        random_state,
        memory,
        memory_warm_start,
        verbosity,
    ):
        self.objective_function = objective_function
        self.s_space = s_space
        self.n_iter = n_iter

        self.initialize = initialize
        self.pass_through = pass_through
        self.max_score = max_score
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.verbosity = verbosity

        if "progress_bar" in self.verbosity:
            self.verbosity = ["progress_bar"]
        else:
            self.verbosity = []

    def _setup_process(self, nth_process):
        self.nth_process = nth_process

        self.hg_conv = HyperGradientConv(self.s_space)

        initialize = self.hg_conv.conv_initialize(self.initialize)
        search_space_positions = self.s_space.positions

        # conv warm start for smbo from values into positions
        if "warm_start_smbo" in self.opt_params:
            self.opt_params["warm_start_smbo"] = self.hg_conv.conv_memory_warm_start(
                self.opt_params["warm_start_smbo"]
            )

        self._optimizer = self._OptimizerClass(
            search_space=search_space_positions,
            initialize=initialize,
            random_state=self.random_state,
            nth_process=nth_process,
            **self.opt_params
        )

        self.conv = self._optimizer.conv

    def search(self, nth_process):
        self._setup_process(nth_process)

        gfo_wrapper_model = ObjectiveFunction(
            self.objective_function, self._optimizer, self.nth_process
        )
        gfo_wrapper_model.pass_through = self.pass_through

        memory_warm_start = self._convert_args2gfo(self.memory_warm_start)

        gfo_objective_function = gfo_wrapper_model(self.s_space())

        self._optimizer.search(
            objective_function=gfo_objective_function,
            n_iter=self.n_iter,
            max_time=self.max_time,
            max_score=self.max_score,
            early_stopping=self.early_stopping,
            memory=self.memory,
            memory_warm_start=memory_warm_start,
            verbosity=self.verbosity,
        )

        self._convert_results2hyper()
        self.p_bar = self._optimizer.p_bar
