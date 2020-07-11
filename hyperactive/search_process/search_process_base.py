# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random
import numpy as np
import pandas as pd

from importlib import import_module


optimizer_dict = {
    "HillClimbing": "HillClimbingOptimizer",
    "StochasticHillClimbing": "StochasticHillClimbingOptimizer",
    "TabuSearch": "TabuOptimizer",
    "RandomSearch": "RandomSearchOptimizer",
    "RandomRestartHillClimbing": "RandomRestartHillClimbingOptimizer",
    "RandomAnnealing": "RandomAnnealingOptimizer",
    "SimulatedAnnealing": "SimulatedAnnealingOptimizer",
    "StochasticTunneling": "StochasticTunnelingOptimizer",
    "ParallelTempering": "ParallelTemperingOptimizer",
    "ParticleSwarm": "ParticleSwarmOptimizer",
    "EvolutionStrategy": "EvolutionStrategyOptimizer",
    "Bayesian": "BayesianOptimizer",
    "TPE": "TreeStructuredParzenEstimators",
    "DecisionTree": "DecisionTreeOptimizer",
}


class SearchProcess:
    def __init__(
        self,
        nth_process,
        verb,
        objective_function,
        search_space,
        n_iter,
        function_parameter,
        optimizer,
        n_jobs,
        init_para,
        memory,
        hyperactive,
        random_state,
    ):
        self.nth_process = nth_process
        self.verb = verb
        self.objective_function = objective_function
        self.search_space = search_space
        self.n_iter = n_iter
        self.function_parameter = function_parameter
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.init_para = init_para
        self.memory = memory
        self.hyperactive = hyperactive
        self.random_state = random_state

        self._process_arguments()

        self.iter_times = []
        self.eval_times = []

        module = import_module("gradient_free_optimizers")
        self.opt_class = getattr(module, optimizer_dict[optimizer])

        self.res = ResultsManager(objective_function, search_space, function_parameter)

    def _results_dict(self):
        results_dict = {
            "eval_times": self.eval_times,
            "iter_times": self.iter_times,
            "memory": self.cand.memory_dict_new,
            "para_best": self.cand.para_best,
            "score_best": self.cand.score_best,
        }

        return results_dict

    def _time_exceeded(self, start_time, max_time):
        run_time = time.time() - start_time
        return max_time and run_time > max_time

    def _initialize_search(self, nth_process):
        init_positions = self.cand.init.set_start_pos(self.n_positions)
        self.opt = self.opt_class(init_positions, self.cand.space.dim, opt_para={})

        self.verb.p_bar.init_p_bar(nth_process, self.n_iter, self.objective_function)

    def _process_arguments(self):
        self._set_random_seed()

        if isinstance(self.optimizer, dict):
            optimizer = list(self.optimizer.keys())[0]
            self.opt_para = self.optimizer[optimizer]
            self.optimizer = optimizer

            self.n_positions = self._get_n_positions()
        else:
            self.opt_para = {}
            self.n_positions = self._get_n_positions()

    def _get_n_positions(self):
        n_positions_strings = [
            "n_positions",
            "system_temperatures",
            "n_particles",
            "individuals",
        ]

        n_positions = 1
        for n_pos_name in n_positions_strings:
            if n_pos_name in list(self.opt_para.keys()):
                n_positions = self.opt_para[n_pos_name]
                if n_positions == "system_temperatures":
                    n_positions = len(n_positions)

        return n_positions

    def _set_random_seed(self):
        """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
        if self.random_state is None:
            self.random_state = np.random.randint(0, high=2 ** 32 - 2)

        random.seed(self.random_state + self.nth_process)
        np.random.seed(self.random_state + self.nth_process)

    def store_memory(self, memory):
        pass

    def print_best_para(self):
        self.verb.info.print_start_point()

    def search(self, start_time, max_time, nth_process):
        self._initialize_search(nth_process)

        # loop to initialize N positions
        for nth_init in range(len(self.opt.init_positions)):
            start_time = time.time()
            pos_new = self.opt.init_pos(nth_init)

            start_time = time.time()
            score_new = self.cand.get_score(pos_new, 0)
            self.eval_times.append(time.time() - start_time)

            self.opt.evaluate(score_new)
            self.iter_times.append(time.time() - start_time)

        # loop to do the iterations
        for nth_iter in range(len(self.opt.init_positions), self.n_iter):
            start_time = time.time()
            pos_new = self.opt.iterate(nth_iter)

            start_time = time.time()
            score_new = self.cand.get_score(pos_new, nth_iter)
            self.eval_times.append(time.time() - start_time)

            self.opt.evaluate(score_new)
            self.iter_times.append(time.time() - start_time)

            if self._time_exceeded(start_time, max_time):
                break

        self.verb.p_bar.close_p_bar()

        self.res.memory_dict_new = self.cand.memory_dict_new
        self.res.results_dict = self._results_dict()

        return self.res


from hypermemory import HyperactiveWrapper
from ..meta_data.meta_data_path import meta_data_path


class ResultsManager:
    def __init__(
        self, objective_function, search_space, function_parameter,
    ):
        self.objective_function = objective_function
        self.search_space = search_space
        self.function_parameter = function_parameter

        self.memory_dict_new = {}

        self.hypermem = HyperactiveWrapper(
            main_path=meta_data_path(),
            X=function_parameter["features"],
            y=function_parameter["target"],
            model=self.objective_function,
            search_space=search_space,
        )

    def load_long_term_memory(self):
        return self.hypermem.load()

    def save_long_term_memory(self):
        self.hypermem.save(self.memory_dict_new)
