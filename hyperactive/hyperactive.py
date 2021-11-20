# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import multiprocessing as mp
from tqdm import tqdm

from .optimizers import RandomSearchOptimizer
from .run_search import run_search
from .print_info import print_info

from .hyperactive_results import HyperactiveResults
from .search_space import SearchSpace


class Hyperactive(HyperactiveResults):
    def __init__(
        self,
        verbosity=["progress_bar", "print_results", "print_times"],
        distribution="multiprocessing",
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

        self.progress_boards = {}

    def _create_shared_memory(self, memory, objective_function, optimizer):
        if memory == "share":
            if len(self.process_infos) == 0:
                manager = mp.Manager()
                memory = manager.dict()

            for process_info in self.process_infos.values():
                same_obj_func = (
                    process_info["objective_function"].__name__
                    == objective_function.__name__
                )
                same_ss_length = len(process_info["optimizer"].s_space()) == len(
                    optimizer.s_space()
                )

                if same_obj_func and same_ss_length:
                    memory = process_info["memory"]
                else:
                    manager = mp.Manager()
                    memory = manager.dict()

        return memory

    def _add_search_processes(
        self,
        random_state,
        objective_function,
        optimizer,
        n_iter,
        n_jobs,
        max_score,
        early_stopping,
        memory,
        memory_warm_start,
        search_id,
    ):
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        for _ in range(n_jobs):
            nth_process = len(self.process_infos)
            memory = self._create_shared_memory(memory, objective_function, optimizer)

            self.process_infos[nth_process] = {
                "random_state": random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "objective_function": objective_function,
                "optimizer": optimizer,
                "n_iter": n_iter,
                "max_score": max_score,
                "early_stopping": early_stopping,
                "memory": memory,
                "memory_warm_start": memory_warm_start,
                "search_id": search_id,
            }

    @staticmethod
    def _default_opt(optimizer):
        if isinstance(optimizer, str):
            if optimizer == "default":
                optimizer = RandomSearchOptimizer()
        return optimizer

    @staticmethod
    def _default_search_id(search_id, objective_function):
        if not search_id:
            search_id = objective_function.__name__
        return search_id

    def _init_progress_board(self, progress_board, search_id, search_space):
        if progress_board:
            data_c = progress_board.init_paths(search_id, search_space)

            if progress_board.uuid not in self.progress_boards:
                self.progress_boards[progress_board.uuid] = progress_board

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
        early_stopping=None,
        random_state=None,
        memory=True,
        memory_warm_start=None,
        progress_board=None,
    ):
        optimizer = self._default_opt(optimizer)
        search_id = self._default_search_id(search_id, objective_function)
        progress_collector = self._init_progress_board(
            progress_board, search_id, search_space
        )

        s_space = SearchSpace(search_space)
        optimizer.init(s_space, initialize, progress_collector)

        self._add_search_processes(
            random_state,
            objective_function,
            optimizer,
            n_iter,
            n_jobs,
            max_score,
            early_stopping,
            memory,
            memory_warm_start,
            search_id,
        )

    def _print_info(self):
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

    def run(self, max_time=None, _test_st_backend=False):
        for nth_process in self.process_infos.keys():
            self.process_infos[nth_process]["max_time"] = max_time

        # open progress board
        if not _test_st_backend:
            for progress_board in self.progress_boards.values():
                progress_board.open_dashboard()

        self.results_list = run_search(
            self.process_infos, self.distribution, self.n_processes
        )

        # delete lock files
        if not _test_st_backend:
            for progress_board in self.progress_boards.values():
                for search_id in progress_board.search_ids:
                    progress_board._io_.remove_lock(search_id)

        self._print_info()
