# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import multiprocessing
from .optimizers import RandomSearchOptimizer

from .run_search import run_search


def set_n_jobs(n_jobs):
    """Sets the number of jobs to run in parallel"""
    num_cores = multiprocessing.cpu_count()
    if n_jobs == -1 or n_jobs > num_cores:
        return num_cores
    else:
        return n_jobs


class Hyperactive:
    def __init__(
        self,
        verbosity={
            "progress_bar": True,
            "print_results": True,
            "print_times": True,
        },
        distribution="multiprocessing",
    ):
        self.verbosity = verbosity
        self.distribution = distribution

        self.process_infos = {}

    def _add_search_processes(self):
        for nth_job in range(set_n_jobs(self.n_jobs)):
            nth_process = len(self.process_infos)

            self.process_infos[nth_process] = {
                "random_state": self.random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "objective_function": self.objective_function,
                "search_space": self.search_space,
                "optimizer": self.optimizer,
                "n_iter": self.n_iter,
                "initialize": self.initialize,
                "max_score": self.max_score,
                "memory": self.memory,
            }

    def add_search(
        self,
        objective_function,
        search_space,
        n_iter,
        optimizer=RandomSearchOptimizer(),
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        max_score=None,
        random_state=None,
        memory=True,
    ):
        self.objective_function = objective_function
        self.search_space = search_space
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.initialize = initialize
        self.max_score = max_score
        self.random_state = random_state
        self.memory = memory

        self.optimizer.init(search_space)

        self._add_search_processes()

    def run(
        self, max_time=None,
    ):
        for nth_process in self.process_infos.keys():
            self.process_infos[nth_process]["max_time"] = max_time

        results_list = run_search(self.process_infos, self.distribution)

        self.best_score = {}
        self.best_para = {}
        self.results = {}

        for results in results_list:
            nth_process = results["nth_process"]

            process_infos = self.process_infos[nth_process]
            objective_function = process_infos["objective_function"]

            self.best_score[objective_function] = results["best_score"]
            self.best_para[objective_function] = results["best_para"]
            self.results[objective_function] = results["results"]

