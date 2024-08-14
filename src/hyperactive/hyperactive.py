# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import copy
import multiprocessing as mp
import pandas as pd

from typing import Union, List, Dict, Type

from .optimizers import RandomSearchOptimizer
from .run_search import run_search

from .results import Results
from .print_results import PrintResults
from .search_space import SearchSpace


class Hyperactive:
    def __init__(
        self,
        verbosity: list = ["progress_bar", "print_results", "print_times"],
        distribution: str = "multiprocessing",
        n_processes: Union[str, int] = "auto",
    ):
        super().__init__()
        if verbosity is False:
            verbosity = []

        self.verbosity = verbosity
        self.distribution = distribution
        self.n_processes = n_processes

        self.opt_pros = {}

    def _create_shared_memory(self):
        _bundle_opt_processes = {}

        for opt_pros in self.opt_pros.values():
            if opt_pros.memory != "share":
                continue
            name = opt_pros.objective_function.__name__

            if name not in _bundle_opt_processes:
                _bundle_opt_processes[name] = [opt_pros]
            else:
                _bundle_opt_processes[name].append(opt_pros)

        for opt_pros_l in _bundle_opt_processes.values():
            for idx, opt_pros in enumerate(opt_pros_l):
                ss_equal = len(opt_pros.s_space()) == len(opt_pros_l[0].s_space())

                if idx == 0 or not ss_equal:
                    manager = mp.Manager()  # get new manager.dict
                    opt_pros.memory = manager.dict()
                else:
                    opt_pros.memory = opt_pros_l[0].memory  # get same manager.dict

    @staticmethod
    def _default_opt(optimizer):
        if isinstance(optimizer, str):
            if optimizer == "default":
                optimizer = RandomSearchOptimizer()
        return copy.deepcopy(optimizer)

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
        objective_function: callable,
        search_space: Dict[str, list],
        n_iter: int,
        search_id=None,
        optimizer: Union[str, Type[RandomSearchOptimizer]] = "default",
        n_jobs: int = 1,
        initialize: Dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[callable] = None,
        pass_through: Dict = None,
        callbacks: Dict[str, callable] = None,
        catch: Dict = None,
        max_score: float = None,
        early_stopping: Dict = None,
        random_state: int = None,
        memory: Union[str, bool] = "share",
        memory_warm_start: pd.DataFrame = None,
    ):
        self.check_list(search_space)

        if constraints is None:
            constraints = []
        if pass_through is None:
            pass_through = {}
        if callbacks is None:
            callbacks = {}
        if catch is None:
            catch = {}
        if early_stopping is None:
            early_stopping = {}

        optimizer = self._default_opt(optimizer)
        search_id = self._default_search_id(search_id, objective_function)
        s_space = SearchSpace(search_space)

        optimizer.setup_search(
            objective_function=objective_function,
            s_space=s_space,
            n_iter=n_iter,
            initialize=initialize,
            constraints=constraints,
            pass_through=pass_through,
            callbacks=callbacks,
            catch=catch,
            max_score=max_score,
            early_stopping=early_stopping,
            random_state=random_state,
            memory=memory,
            memory_warm_start=memory_warm_start,
            verbosity=self.verbosity,
        )

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        for _ in range(n_jobs):
            nth_process = len(self.opt_pros)
            self.opt_pros[nth_process] = optimizer

    def _print_info(self):
        print_res = PrintResults(self.opt_pros, self.verbosity)

        if self.verbosity:
            for _ in range(len(self.opt_pros)):
                print("")

        for results in self.results_list:
            nth_process = results["nth_process"]
            print_res.print_process(results, nth_process)

    def run(self, max_time: float = None):
        self._create_shared_memory()

        for opt in self.opt_pros.values():
            opt.max_time = max_time

        self.results_list = run_search(
            self.opt_pros, self.distribution, self.n_processes
        )

        self.results_ = Results(self.results_list, self.opt_pros)

        self._print_info()

    def best_para(self, id_):
        return self.results_.best_para(id_)

    def best_score(self, id_):
        return self.results_.best_score(id_)

    def search_data(self, id_, times=False):
        search_data_ = self.results_.search_data(id_)

        if times == False:
            search_data_.drop(
                labels=["eval_times", "iter_times"],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return search_data_
