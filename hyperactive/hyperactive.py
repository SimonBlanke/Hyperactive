# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from tqdm import tqdm

from .optimizers import RandomSearchOptimizer
from .run_search import run_search
from .print_info import print_info

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

        if not search_id:
            search_id = objective_function.__name__

        optimizer.init(search_space, initialize)

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
