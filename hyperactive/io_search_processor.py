# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from importlib import import_module

import time
import random
import numpy as np
import pandas as pd

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
    "TreeStructured": "TreeStructuredParzenEstimators",
    "DecisionTree": "DecisionTreeOptimizer",
}


class IoSearchProcessor:
    def __init__(
        self,
        nth_process,
        p_bar,
        model,
        search_space,
        n_iter,
        optimizer,
        init_para,
        random_state,
    ):
        self.nth_process = nth_process
        self.p_bar = p_bar
        self.model = model
        self.search_space = search_space
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.init_para = init_para
        self.random_state = random_state

        self._process_arguments()

        module = import_module("gradient_free_optimizers")
        self.opt_class = getattr(module, optimizer_dict[self.optimizer])

    def _process_arguments(self):
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

    def _set_random_seed(self, nth_process):
        """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
        if self.random_state is None:
            self.random_state = np.random.randint(0, high=2 ** 32 - 2)

        random.seed(self.random_state + nth_process)
        np.random.seed(self.random_state + nth_process)

    def _get_optimizer(self, cand):
        init_positions = cand.init.set_start_pos(self.n_positions)
        return self.opt_class(cand.space.search_space_pos, **self.opt_para)

    def init_search(self, nth_process, cand):
        self.p_bar.init_p_bar(nth_process, self.n_iter, self.model)
        self._set_random_seed(nth_process)

        return self._get_optimizer(cand)
