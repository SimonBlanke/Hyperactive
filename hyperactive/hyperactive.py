# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numpy as np
import multiprocessing
from importlib import import_module


from .checks import check_args
from .search import SearchManager
from .process import _process_
from .search_info import SearchInfo

from optimization_metadata import HyperactiveWrapper
from .meta_data.meta_data_path import meta_data_path

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


def set_n_jobs(n_jobs):
    """Sets the number of jobs to run in parallel"""
    num_cores = multiprocessing.cpu_count()
    if n_jobs == -1 or n_jobs > num_cores:
        return num_cores
    else:
        return n_jobs


def get_class(file_path, class_name):
    module = import_module(file_path, "hyperactive")
    return getattr(module, class_name)


def no_ext_warnings():
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn


def init_optimizer(optimizer, search_space):
    if isinstance(optimizer, dict):
        opt_string = list(optimizer.keys())[0]
        opt_para = optimizer[opt_string]
    else:
        opt_string = optimizer
        opt_para = {}

    module = import_module("gradient_free_optimizers")
    opt_class = getattr(module, optimizer_dict[opt_string])

    search_space_gfo = {}
    for key in search_space.keys():
        dict_value = search_space[key]
        space_dim = np.array(range(len(dict_value)))
        # search_space_pos.append(space_dim)
        search_space_gfo[key] = space_dim

    opt = opt_class(search_space_gfo, **opt_para)

    return opt


class Hyperactive:
    def __init__(
        self,
        X,
        y,
        random_state=None,
        verbosity={
            "print_search_info": True,
            "progress_bar": True,
            "print_results": True,
        },
        ext_warnings=False,
    ):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.verbosity = verbosity

        self.s_info = SearchInfo()

        self.opt_metadata_dict = {}
        self.model_memory_dict = {}

        self.process_info_dict = {}
        self.search_process_dict = {}

    def add_search(
        self,
        model,
        search_space,
        n_iter,
        name=None,
        optimizer="RandomSearch",
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        memory="short",
    ):
        """
        check_args(
            model, search_space, n_iter, name, optimizer, n_jobs, initialize, memory,
        )
        """
        n_jobs = set_n_jobs(n_jobs)

        if model not in self.opt_metadata_dict:
            self.opt_metadata_dict[model] = HyperactiveWrapper(
                main_path=meta_data_path(),
                X=self.X,
                y=self.y,
                model=model,
                search_space=search_space,
                verbosity=0,
            )

            self.model_memory_dict[model] = self.opt_metadata_dict[model].load()

            memory_dict = self.model_memory_dict[model]
            values = np.array(list(memory_dict.keys()))
            scores = np.array(list(memory_dict.values())).reshape(-1,)

        processes = []
        for nth_job in range(n_jobs):
            nth_process = len(self.process_info_dict)
            processes.append(nth_process)

            self.process_info_dict[nth_process] = {
                "X": self.X,
                "y": self.y,
                "random_state": self.random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "model": model,
                "search_space": search_space,
                "n_iter": n_iter,
                "name": name,
                "optimizer": init_optimizer(optimizer, search_space),
                "initialize": initialize,
                "memory": True,
            }

        self.s_info.add_search(
            processes,
            model,
            search_space,
            n_iter,
            name,
            optimizer,
            n_jobs,
            initialize,
            memory,
            self.model_memory_dict[model],
        )

    def run(self, max_time=None, distribution=None):
        for key in self.process_info_dict.keys():
            self.process_info_dict[key]["max_time"] = max_time
            self.process_info_dict[key]["distribution"] = distribution

        self.s_manage = SearchManager(self.s_info)
        self.s_manage.run(self.process_info_dict)

        for results in self.s_manage.results_list:
            nth_process = results["nth_process"]
            memory_dict_new = results["memory_dict_new"]

            model = self.process_info_dict[nth_process]["model"]
            metadata = self.opt_metadata_dict[model]

            metadata.save(results["memory_dict_new"])

        """
        self.eval_times = self.search.eval_times_dict
        self.iter_times = self.search.iter_times_dict

        self.positions = self.search.positions_dict
        self.scores = self.search.scores_dict
        self.score_best_list = self.search.best_score_list_dict

        self.para_best = self.search.para_best_dict
        self.score_best = self.search.score_best_dict

        self.position_results = self.search.position_results
        """

