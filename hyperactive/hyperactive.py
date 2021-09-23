# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import multiprocessing as mp
from tqdm import tqdm

from .optimizers import RandomSearchOptimizer
from .run_search import run_search
from .print_info import print_info

from .results import Results


class Hyperactive:
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
        if memory is not False:
            if len(self.process_infos) == 0:
                manager = mp.Manager()
                memory = manager.dict()

            for process_info in self.process_infos.values():
                same_obj_func = process_info["objective_function"] == objective_function
                same_ss_length = len(process_info["optimizer"].search_space) == len(
                    optimizer.search_space
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
        search_space,
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
            if memory == "share":
                memory = self._create_shared_memory(
                    memory, objective_function, optimizer
                )

            self.process_infos[nth_process] = {
                "random_state": random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "objective_function": objective_function,
                "search_space": search_space,
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

    @staticmethod
    def check_list(search_space):
        for key in search_space.keys():
            search_dim = search_space[key]

            error_msg = (
                "Value in '{}' of search space dictionary must be of type list".format(
                    key
                )
            )
            if not isinstance(search_dim, list):
                print("Warning", error_msg)
                # raise ValueError(error_msg)

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
        memory="share",
        memory_warm_start=None,
    ):
        optimizer = self._default_opt(optimizer)
        search_id = self._default_search_id(search_id, objective_function)

        self.check_list(search_space)

        optimizer.init(search_space, initialize)

        self._add_search_processes(
            random_state,
            objective_function,
            search_space,
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

    def run(self, max_time=None):
        for nth_process in self.process_infos.keys():
            self.process_infos[nth_process]["max_time"] = max_time

        self.results_list = run_search(
            self.process_infos, self.distribution, self.n_processes
        )

        self.results_ = Results(self.results_list, self.process_infos)

        self._print_info()

    def best_para(self, id_):
        return self.results_.best_para(id_)

    def best_score(self, id_):
        return self.results_.best_score(id_)

    def search_data(self, id_):
        return self.results_.search_data(id_)
