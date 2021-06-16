# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
from tqdm import tqdm

from .optimizers import RandomSearchOptimizer
from .run_search import run_search
from .print_info import print_info
from .data_tools import DataCollector

from .hyperactive_results import HyperactiveResults


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
        n_processes="auto",
    ):
        super().__init__()
        if verbosity is False:
            verbosity = []

        self.verbosity = verbosity
        self.distribution = distribution
        self.n_processes = n_processes

        self.search_ids = []
        self.process_infos = {}
        self.objFunc2results = {}
        self.search_id2results = {}

        self.progress_paths = []

        self.progress_boards = {}

    def _add_search_processes(
        self,
        random_state,
        objective_function,
        optimizer,
        n_iter,
        n_jobs,
        max_score,
        memory,
        memory_warm_start,
        search_id,
    ):
        for _ in range(n_jobs):
            nth_process = len(self.process_infos)

            self.process_infos[nth_process] = {
                "random_state": random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "objective_function": objective_function,
                "optimizer": optimizer,
                "n_iter": n_iter,
                "max_score": max_score,
                "memory": memory,
                "memory_warm_start": memory_warm_start,
                "search_id": search_id,
            }

    def _default_opt(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "default":
                optimizer = RandomSearchOptimizer()
        return optimizer

    def _default_search_id(self, search_id, objective_function):
        if not search_id:
            search_id = objective_function.__name__
        return search_id

    def _create_filter_file(self, search_id, search_space):
        import numpy as np
        import pandas as pd

        filter_path = "./filter_" + search_id + ".csv"
        if os.path.isfile(filter_path):
            os.remove(filter_path)

        indices = list(search_space.keys()) + ["score"]
        filter_dict = {
            "parameter": indices,
            "lower bound": "lower",
            "upper bound": "upper",
        }

        df = pd.DataFrame(filter_dict)
        df.to_csv(filter_path, index=None)

    def _init_progress_board(self, progress_board, search_id):
        if progress_board:
            progress_data_path = "./progress_data_" + search_id + ".csv~"

            if os.path.isfile(progress_data_path):
                os.remove(progress_data_path)

            data_c = DataCollector(progress_data_path)
            # self.progress_paths.append(progress_data_path)

            progress_board.paths_list.append(search_id)
            if progress_board.uuid not in self.progress_boards:
                self.progress_boards[progress_board.uuid] = progress_board
        else:
            data_c = None

        return data_c

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
        progress_board=None,
    ):
        optimizer = self._default_opt(optimizer)
        search_id = self._default_search_id(search_id, objective_function)

        data_c = self._init_progress_board(progress_board, search_id)
        self._create_filter_file(search_id, search_space)

        optimizer.init(search_space, initialize, data_c)

        self._add_search_processes(
            random_state,
            objective_function,
            optimizer,
            n_iter,
            n_jobs,
            max_score,
            memory,
            memory_warm_start,
            search_id,
        )

    def run(self, max_time=None):
        for nth_process in self.process_infos.keys():
            self.process_infos[nth_process]["max_time"] = max_time

        # open progress board
        for progress_board in self.progress_boards.values():
            progress_board.open_dashboard()

        self.results_list = run_search(
            self.process_infos, self.distribution, self.n_processes
        )

        for results in self.results_list:
            nth_process = results["nth_process"]

            print_info(
                verbosity=self.process_infos[nth_process]["verbosity"],
                objective_function=self.process_infos[nth_process][
                    "objective_function"
                ],
                best_score=results["best_score"],
                best_para=results["best_para"],
                best_iter=results["best_iter"],
                eval_times=results["eval_times"],
                iter_times=results["iter_times"],
                n_iter=self.process_infos[nth_process]["n_iter"],
            )
