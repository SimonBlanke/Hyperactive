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
    """
    Initialize the Hyperactive class to manage optimization processes.

    Parameters:
    - verbosity: List of verbosity levels (default: ["progress_bar", "print_results", "print_times"])
    - distribution: String indicating the distribution method (default: "multiprocessing")
    - n_processes: Number of processes to run in parallel or "auto" to determine automatically (default: "auto")

    Methods:
    - add_search: Add a new optimization search process with specified parameters
    - run: Execute the optimization searches
    - best_para: Get the best parameters for a specific search
    - best_score: Get the best score for a specific search
    - search_data: Get the search data for a specific search
    """

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

            _bundle_opt_processes.setdefault(name, []).append(opt_pros)

        for opt_pros_l in _bundle_opt_processes.values():
            # Check if the lengths of the search spaces of all optimizers in the list are the same.
            if (
                len(set(len(opt_pros.s_space()) for opt_pros in opt_pros_l))
                == 1
            ):
                manager = mp.Manager()  # get new manager.dict
                shared_memory = manager.dict()
                for opt_pros in opt_pros_l:
                    opt_pros.memory = shared_memory
            else:
                for opt_pros in opt_pros_l:
                    opt_pros.memory = opt_pros_l[
                        0
                    ].memory  # get same manager.dict

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

            error_msg = "Value in '{}' of search space dictionary must be of type list".format(
                key
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
        """
        Add a new optimization search process with specified parameters.

        Parameters:
        - objective_function: The objective function to optimize.
        - search_space: Dictionary defining the search space for optimization.
        - n_iter: Number of iterations for the optimization process.
        - search_id: Identifier for the search process (default: None).
        - optimizer: The optimizer to use for the search process (default: "default").
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

        self.check_list(search_space)

        constraints = constraints or []
        pass_through = pass_through or {}
        callbacks = callbacks or {}
        catch = catch or {}
        early_stopping = early_stopping or {}

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

        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

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
        """
        Run the optimization process with an optional maximum time limit.

        Args:
            max_time (float, optional): Maximum time limit for the optimization process. Defaults to None.
        """

        self._create_shared_memory()

        for opt in self.opt_pros.values():
            opt.max_time = max_time

        self.results_list = run_search(
            self.opt_pros, self.distribution, self.n_processes
        )

        self.results_ = Results(self.results_list, self.opt_pros)

        self._print_info()

    def best_para(self, id_):
        """
        Retrieve the best parameters for a specific ID from the results.

        Parameters:
        - id_ (int): The ID of the parameters to retrieve.

        Returns:
        - Union[Dict[str, Union[int, float]], None]: The best parameters for the specified ID if found, otherwise None.

        Raises:
        - ValueError: If the objective function name is not recognized.
        """

        return self.results_.best_para(id_)

    def best_score(self, id_):
        """
        Return the best score for a specific ID from the results.

        Parameters:
        - id_ (int): The ID for which the best score is requested.
        """

        return self.results_.best_score(id_)

    def search_data(self, id_, times=False):
        """
        Retrieve search data for a specific ID from the results. Optionally exclude evaluation and iteration times if 'times' is set to False.

        Parameters:
        - id_ (int): The ID of the search data to retrieve.
        - times (bool, optional): Whether to exclude evaluation and iteration times. Defaults to False.

        Returns:
        - pd.DataFrame: The search data for the specified ID.
        """

        search_data_ = self.results_.search_data(id_)

        if times == False:
            search_data_.drop(
                labels=["eval_times", "iter_times"],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return search_data_
