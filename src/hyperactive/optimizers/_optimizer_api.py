"""Base class for optimizer."""

import numpy as np
from typing import Union, List, Dict, Type
import copy
import multiprocessing as mp
import pandas as pd

from .backend_stuff.search_space import SearchSpace
from .backend_stuff.run_search import run_search
from .hyper_optimizer import HyperOptimizer
from .backend_stuff.results import Results
from .backend_stuff.print_results import PrintResults

from skbase.base import BaseObject


class BaseOptimizer(BaseObject):
    """Base class for optimizer."""

    opt_pros = {}

    def __init__(self, optimizer_class, opt_params):
        super().__init__()
        self.opt_params = opt_params
        self.hyper_optimizer = HyperOptimizer(optimizer_class, opt_params)

    @staticmethod
    def _default_search_id(search_id, objective_function):
        if not search_id:
            search_id = objective_function.__name__
        return search_id

    @staticmethod
    def check_list(search_space):
        for key in search_space.keys():
            search_dim = search_space[key]

            error_msg = "Value in '{}' of search space dictionary must be of type list".format(
                key
            )
            if not isinstance(search_dim, list):
                print("Warning", error_msg)
                # raise ValueError(error_msg)

    def add_search(
        self,
        experiment: callable,
        search_space: Dict[str, list],
        n_iter: int,
        search_id=None,
        n_jobs: int = 1,
        verbosity: list = ["progress_bar", "print_results", "print_times"],
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
        """
        Add a new optimization search process with specified parameters.

        Parameters:
        - objective_function: The objective function to optimize.
        - search_space: Dictionary defining the search space for optimization.
        - n_iter: Number of iterations for the optimization process.
        - search_id: Identifier for the search process (default: None).
        - n_jobs: Number of parallel jobs to run (default: 1).
        - initialize: Dictionary specifying initialization parameters (default: {"grid": 4, "random": 2, "vertices": 4}).
        - constraints: List of constraint functions (default: None).
        - pass_through: Dictionary of additional parameters to pass through (default: None).
        - callbacks: Dictionary of callback functions (default: None).
        - catch: Dictionary of exceptions to catch during optimization (default: None).
        - max_score: Maximum score to achieve (default: None).
        - early_stopping: Dictionary specifying early stopping criteria (default: None).
        - random_state: Seed for random number generation (default: None).
        - memory: Option to share memory between processes (default: "share").
        - memory_warm_start: DataFrame containing warm start memory (default: None).
        """

        objective_function = experiment._score

        self.check_list(search_space)

        constraints = constraints or []
        pass_through = pass_through or {}
        callbacks = callbacks or {}
        catch = catch or {}
        early_stopping = early_stopping or {}

        search_id = self._default_search_id(search_id, objective_function)
        s_space = SearchSpace(search_space)
        self.verbosity = verbosity

        self.hyper_optimizer.setup_search(
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
            verbosity=verbosity,
        )

        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        for _ in range(n_jobs):
            nth_process = len(self.opt_pros)
            self.opt_pros[nth_process] = self.hyper_optimizer

    def _print_info(self):
        print_res = PrintResults(self.opt_pros, self.verbosity)

        if self.verbosity:
            for _ in range(len(self.opt_pros)):
                print("")

        for results in self.results_list:
            nth_process = results["nth_process"]
            print_res.print_process(results, nth_process)

    def run(
        self,
        max_time=None,
        distribution: str = "multiprocessing",
        n_processes: Union[str, int] = "auto",
    ):
        for opt in self.opt_pros.values():
            opt.max_time = max_time

        self.results_list = run_search(self.opt_pros, distribution, n_processes)

        self.results_ = Results(self.results_list, self.opt_pros)

        self._print_info()
