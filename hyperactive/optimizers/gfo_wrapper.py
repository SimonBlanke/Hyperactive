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
            values_list = self.search_space[para_name]
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
            subset=self.trafo.para_names, keep="first"
        )
        self.memory_values_df = results_dd[
            self.trafo.para_names + ["score"]
        ].reset_index(drop=True)


class _BaseOptimizer_(TrafoClass):
    def __init__(self, **opt_params):
        super().__init__()
        self.opt_params = opt_params

    def init(self, search_space, initialize, progress_collector):
        self.search_space = search_space
        self.initialize = initialize
        self.progress_collector = progress_collector

        self.trafo = HyperGradientTrafo(search_space)

        initialize = self.trafo.trafo_initialize(initialize)
        search_space_positions = self.trafo.search_space_positions

        # trafo warm start for smbo from values into positions
        if "warm_start_smbo" in self.opt_params:
            self.opt_params["warm_start_smbo"] = self.trafo.trafo_memory_warm_start(
                self.opt_params["warm_start_smbo"]
            )

        self._optimizer = self._OptimizerClass(
            search_space_positions, initialize, **self.opt_params
        )

        self.conv = self._optimizer.conv

    def check_LTM(self, memory):
        try:
            memory.study_id
            memory.model_id
        except:
            self.memory = memory
        else:
            self.init_ltm(memory)

    def init_ltm(self, memory):
        self.ltm = copy.deepcopy(memory)
        self.ltm.init_study(
            self.objective_function, self.search_space, self.nth_process
        )
        self.memory_warm_start = self.ltm.load()
        self.memory = True

        print("\n self.memory_warm_start \n", self.memory_warm_start)

    def search(
        self,
        objective_function,
        n_iter,
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
        self.objective_function = objective_function
        self.nth_process = nth_process

        gfo_wrapper_model = ObjectiveFunction(
            objective_function, self._optimizer, nth_process
        )

        # ltm init
        self.check_LTM(memory)
        memory_warm_start = self._convert_args2gfo(memory_warm_start)

        self._optimizer.search(
            gfo_wrapper_model(self.search_space, self.progress_collector),
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
        self.p_bar = self._optimizer.p_bar

        # ltm save after finish
        """
        if inspect.isclass(type(memory)):
            self.ltm.save_on_finish(self.results)
        """
