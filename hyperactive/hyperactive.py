# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm

from .optimizers import RandomSearchOptimizer
from .run_search import run_search


def set_n_jobs(n_jobs):
    """Sets the number of jobs to run in parallel"""
    num_cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        return num_cores
    else:
        return n_jobs


class HyperactiveResults:
    def __init__(*args, **kwargs):
        pass

    def _sort_results_objFunc(self, objective_function):
        best_score = -np.inf
        best_para = None
        search_data = None

        results_list = []

        for results_ in self.results_list:
            nth_process = results_["nth_process"]

            process_infos = self.process_infos[nth_process]
            objective_function_ = process_infos["objective_function"]

            if objective_function_ != objective_function:
                continue

            if results_["best_score"] > best_score:
                best_score = results_["best_score"]
                best_para = results_["best_para"]

            results_list.append(results_["results"])

        if len(results_list) > 0:
            search_data = pd.concat(results_list)

        self.objFunc2results[objective_function] = {
            "best_para": best_para,
            "best_score": best_score,
            "search_data": search_data,
        }

    def _sort_results_search_id(self, search_id):
        for results_ in self.results_list:
            nth_process = results_["nth_process"]
            search_id_ = self.process_infos[nth_process]["search_id"]

            if search_id_ != search_id:
                continue

            best_score = results_["best_score"]
            best_para = results_["best_para"]
            search_data = results_["results"]

            self.search_id2results[search_id] = {
                "best_para": best_para,
                "best_score": best_score,
                "search_data": search_data,
            }

    def _get_one_result(self, id_, result_name):
        if isinstance(id_, str):
            if id_ not in self.search_id2results:
                self._sort_results_search_id(id_)

            return self.search_id2results[id_][result_name]

        else:
            if id_ not in self.objFunc2results:
                self._sort_results_objFunc(id_)

            return self.objFunc2results[id_][result_name]

    def best_para(self, id_):
        return self._get_one_result(id_, "best_para")

    def best_score(self, id_):
        return self._get_one_result(id_, "best_score")

    def results(self, id_):
        return self._get_one_result(id_, "search_data")


class Hyperactive(HyperactiveResults):
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
        super().__init__()
        if verbosity is False:
            verbosity = []

        self.verbosity = verbosity
        self.distribution = distribution
        self.search_ids = []

        self.process_infos = {}

        self.objFunc2results = {}
        self.search_id2results = {}

    def _add_search_processes(
        self,
        random_state,
        objective_function,
        search_space,
        optimizer,
        n_iter,
        n_jobs,
        max_score,
        memory,
        memory_warm_start,
        search_id,
    ):
        for nth_job in range(set_n_jobs(n_jobs)):
            nth_process = len(self.process_infos)

            self.process_infos[nth_process] = {
                "random_state": random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "objective_function": objective_function,
                "search_space": search_space,
                "optimizer": optimizer,
                "n_iter": n_iter,
                "max_score": max_score,
                "memory": memory,
                "memory_warm_start": memory_warm_start,
                "search_id": search_id,
            }

    def add_search(
        self,
        objective_function,
        search_space,
        n_iter,
        search_id=None,
        optimizer="default",
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        max_score=None,
        random_state=None,
        memory=True,
        memory_warm_start=None,
    ):

        if isinstance(optimizer, str):
            if optimizer == "default":
                optimizer = RandomSearchOptimizer()
        optimizer.init(search_space, initialize)

        if search_id is not None:
            search_id = search_id
            self.search_ids.append(search_id)
        else:
            search_id = str(len(self.search_ids))
            self.search_ids.append(search_id)

        self._add_search_processes(
            random_state,
            objective_function,
            search_space,
            optimizer,
            n_iter,
            n_jobs,
            max_score,
            memory,
            memory_warm_start,
            search_id,
        )

    def run(
        self,
        max_time=None,
    ):
        for nth_process in self.process_infos.keys():
            self.process_infos[nth_process]["max_time"] = max_time

        self.results_list = run_search(self.process_infos, self.distribution)
