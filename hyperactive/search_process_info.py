# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from importlib import import_module

optimizer_dict = {
    "HillClimbing": "HillClimbingOptimizer",
    "StochasticHillClimbing": "StochasticHillClimbingOptimizer",
    "TabuSearch": "TabuOptimizer",
    "RandomSearch": "RandomSearchOptimizer",
    "RandomRestartHillClimbing": "RandomRestartHillClimbingOptimizer",
    "RandomAnnealing": "RandomAnnealingOptimizer",
    "SimulatedAnnealing": "SimulatedAnnealingOptimizer",
    "ParallelTempering": "ParallelTemperingOptimizer",
    "ParticleSwarm": "ParticleSwarmOptimizer",
    "EvolutionStrategy": "EvolutionStrategyOptimizer",
    "Bayesian": "BayesianOptimizer",
    "TreeStructured": "TreeStructuredParzenEstimators",
    "DecisionTree": "DecisionTreeOptimizer",
}


class SearchProcessInfo:
    def __init__(self, X, y, random_state, verbosity):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.verbosity = verbosity

        self.process_infos = []

    def _init_optimizer(self, optimizer, search_space):
        if isinstance(optimizer, dict):
            opt_string = list(optimizer.keys())[0]
            opt_para = optimizer[opt_string]
        else:
            opt_string = optimizer
            opt_para = {}

        module = import_module("gradient_free_optimizers")
        opt_class = getattr(module, optimizer_dict[opt_string])

        search_space_pos = []
        for dict_value in search_space.values():
            space_dim = np.array(range(len(dict_value)))
            search_space_pos.append(space_dim)

        opt = opt_class(search_space_pos, **opt_para)

        return opt

    def add_search_process(
        self,
        nth_process,
        model,
        search_space,
        n_iter,
        name,
        optimizer,
        initialize,
        memory,
    ):
        opt = self._init_optimizer(optimizer, search_space)

        self.process_infos.append(
            {
                "nth_process": nth_process,
                "model": model,
                "search_space": search_space,
                "n_iter": n_iter,
                "name": name,
                "optimizer": opt,
                "initialize": initialize,
                "memory": True,
            }
        )

    def add_run_info(self, max_time, distribution):
        for process_info in self.process_infos:
            process_info["max_time"] = max_time
            process_info["distribution"] = distribution
            process_info["X"] = self.X
            process_info["y"] = self.y
            process_info["random_state"] = self.random_state
            process_info["verbosity"] = self.verbosity

