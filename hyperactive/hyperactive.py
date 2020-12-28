# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import multiprocessing
from tqdm import tqdm

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
        verbosity=["progress_bar", "print_results", "print_times"],
        distribution={
            "multiprocessing": {
                "initializer": tqdm.set_lock,
                "initargs": (tqdm.get_lock(),),
            }
        },
    ):
        self.verbosity = verbosity
        self.distribution = distribution
        self.search_ids = []

        self.process_infos = {}

        self.objFunc2results = {}
        self.search_id2results = {}

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
                "memory_warm_start": self.memory_warm_start,
                "search_id": self.search_id,
            }

    def add_search(
        self,
        objective_function,
        search_space,
        n_iter,
        search_id=None,
        optimizer=RandomSearchOptimizer(),
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        max_score=None,
        random_state=None,
        memory=True,
        memory_warm_start=None,
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
        self.memory_warm_start = memory_warm_start

        if search_id is not None:
            self.search_id = search_id
            self.search_ids.append(self.search_id)
        else:
            self.search_id = str(len(self.search_ids))
            self.search_ids.append(self.search_id)

        self.optimizer.init(search_space)

        self._add_search_processes()

    def _sort_results_objFunc(self, objective_function):
        import numpy as np
        import pandas as pd

        best_score = -np.inf
        best_para = None
        results_list = []

        for results in self.results_list:
            nth_process = results["nth_process"]

            process_infos = self.process_infos[nth_process]
            objective_function_ = process_infos["objective_function"]

            if objective_function_ != objective_function:
                continue

            if results["best_score"] > best_score:
                best_score = results["best_score"]
                best_para = results["best_para"]

            results_list.append(results["results"])

        results = pd.concat(results_list)

        self.objFunc2results[objective_function] = {
            "best_para": best_para,
            "best_score": best_score,
            "results": results,
        }

    def _sort_results_search_id(self, search_id):
        for results in self.results_list:
            nth_process = results["nth_process"]
            search_id_ = self.process_infos[nth_process]["search_id"]

            if search_id_ != search_id:
                continue

            best_score = results["best_score"]
            best_para = results["best_para"]
            results = results["results"]

            self.search_id2results[search_id] = {
                "best_para": best_para,
                "best_score": best_score,
                "results": results,
            }

    def run(
        self, max_time=None,
    ):
        for nth_process in self.process_infos.keys():
            self.process_infos[nth_process]["max_time"] = max_time

        self.results_list = run_search(self.process_infos, self.distribution)

    def _get_one_result(self, id, result_name):
        if isinstance(id, str):
            if id not in self.search_id2results:
                self._sort_results_search_id(id)

            return self.search_id2results[id][result_name]

        else:
            if id not in self.objFunc2results:
                self._sort_results_objFunc(id)

            return self.objFunc2results[id][result_name]

    def best_para(self, id):
        return self._get_one_result(id, "best_para")

    def best_score(self, id):
        return self._get_one_result(id, "best_score")

    def results(self, id):
        return self._get_one_result(id, "results")
