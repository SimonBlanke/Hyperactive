"""Base class for optimizer."""

import numpy as np
from typing import Union, List, Dict, Type
import copy
import multiprocessing as mp
import pandas as pd

from .objective_function import ObjectiveFunction
from .hyper_gradient_conv import HyperGradientConv
from .optimizer_attributes import OptimizerAttributes
from .constraint import Constraint
from .backend_stuff.search_space import SearchSpace
from .optimizer_attributes import OptimizerAttributes


from skbase.base import BaseObject


class BaseOptimizer(BaseObject, OptimizerAttributes):
    """Base class for optimizer."""

    def __init__(self, **opt_params):
        super().__init__()
        self.opt_params = opt_params

    def convert_results2hyper(self):
        self.eval_times = sum(self.gfo_optimizer.eval_times)
        self.iter_times = sum(self.gfo_optimizer.iter_times)

        if self.gfo_optimizer.best_para is not None:
            value = self.hg_conv.para2value(self.gfo_optimizer.best_para)
            position = self.hg_conv.position2value(value)
            best_para = self.hg_conv.value2para(position)
            self.best_para = best_para
        else:
            self.best_para = None

        self.best_score = self.gfo_optimizer.best_score
        self.positions = self.gfo_optimizer.search_data
        self.search_data = self.hg_conv.positions2results(self.positions)

        results_dd = self.gfo_optimizer.search_data.drop_duplicates(
            subset=self.s_space.dim_keys, keep="first"
        )
        self.memory_values_df = results_dd[
            self.s_space.dim_keys + ["score"]
        ].reset_index(drop=True)

    def _setup_process(self):
        self.hg_conv = HyperGradientConv(self.s_space)

        search_space_positions = self.s_space.positions

        # conv warm start for smbo from values into positions
        if "warm_start_smbo" in self.opt_params:
            self.opt_params["warm_start_smbo"] = (
                self.hg_conv.conv_memory_warm_start(
                    self.opt_params["warm_start_smbo"]
                )
            )

        self.gfo_optimizer = self.optimizer_class(
            search_space=search_space_positions,
            **self.opt_params,
        )

        self.conv = self.gfo_optimizer.conv

    def add_search(
        self,
        experiment,
        search_config: dict,
        n_iter: int,
        search_id=None,
        n_jobs: int = 1,
        initialize: Dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
    ):
        """Add a new optimization search process with specified parameters.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_config : dict with str keys
            The search configuration dictionary.
        """
        self.experiment = experiment
        self.search_config = search_config
        self.n_iter = n_iter
        self.search_id = search_id
        self.n_jobs = n_jobs
        self.initialize = initialize

        search_id = self._default_search_id(search_id, experiment._score)
        s_space = SearchSpace(search_config.search_space)

        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        for _ in range(n_jobs):
            nth_process = len(self.opt_pros)
            self.opt_pros[nth_process] = optimizer

    def run(
        self,
        max_time=None,
        max_score=None,
        early_stopping=None,
    ):
        self._setup_process()

        gfo_wrapper_model = ObjectiveFunction(
            objective_function=self.experiment._score,
            callbacks=[],
            catch={},
        )

        gfo_objective_function = gfo_wrapper_model(self.s_space())

        self.gfo_optimizer.init_search(
            gfo_objective_function,
            self.n_iter,
            max_time,
            max_score,
            early_stopping,
            False,
        )
        for nth_iter in range(self.n_iter):
            print("iterate")
            self.gfo_optimizer.search_step(nth_iter)
            if self.gfo_optimizer.stop.check():
                break

        self.gfo_optimizer.finish_search()

        self.convert_results2hyper()

        self._add_result_attributes(
            self.best_para,
            self.best_score,
            self.gfo_optimizer.p_bar._best_since_iter,
            self.eval_times,
            self.iter_times,
            self.search_data,
            self.gfo_optimizer.random_seed,
        )
