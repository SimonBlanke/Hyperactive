"""Base class for optimizer."""

from typing import Union, List, Dict
import multiprocessing as mp
import pandas as pd

from .backend_stuff.search_space import SearchSpace
from ._search import Search


from .._composite_optimizer import CompositeOptimizer

from skbase.base import BaseObject


class BaseOptimizer(BaseObject):
    """Base class for optimizer."""

    n_search: int
    searches: list
    opt_pros: dict

    def __init__(self, optimizer_class, opt_params):
        super().__init__()

        self.optimizer_class = optimizer_class
        self.opt_params = opt_params

        self.n_search = 0
        self.searches = []

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
        experiment: callable,
        search_space: Dict[str, list],
        n_iter: int,
        search_id=None,
        n_jobs: int = 1,
        initialize: Dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[callable] = None,
        pass_through: Dict = None,
        max_score: float = None,
        early_stopping: Dict = None,
        random_state: int = None,
        memory: Union[str, bool] = "share",
        memory_warm_start: pd.DataFrame = None,
    ):
        """
        Add a new optimization search process with specified parameters.

        Parameters:
        - experiment: Experiment class containing the objective-function to optimize.
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

        self.n_search += 1

        self.check_list(search_space)

        constraints = constraints or []
        pass_through = pass_through or {}
        early_stopping = early_stopping or {}

        search_id = self._default_search_id(search_id, experiment.objective_function)
        s_space = SearchSpace(search_space)

        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        for _ in range(n_jobs):
            search = Search(self.optimizer_class, self.opt_params)
            search.setup(
                experiment=experiment,
                s_space=s_space,
                n_iter=n_iter,
                initialize=initialize,
                constraints=constraints,
                pass_through=pass_through,
                max_score=max_score,
                early_stopping=early_stopping,
                random_state=random_state,
                memory=memory,
                memory_warm_start=memory_warm_start,
            )
            self.searches.append(search)

    @property
    def nth_search(self):
        return len(self.composite_opt.optimizers)

    def __add__(self, optimizer_instance):
        return CompositeOptimizer(self, optimizer_instance)

    def run(
        self,
        max_time=None,
        distribution: str = "multiprocessing",
        n_processes: Union[str, int] = "auto",
        verbosity: list = ["progress_bar", "print_results", "print_times"],
    ):
        self.comp_opt = CompositeOptimizer(self)
        self.comp_opt.run(max_time, distribution, n_processes, verbosity)

    def best_para(self, experiment):
        """
        Retrieve the best parameters for a specific ID from the results.

        Parameters:
        - experiment (int): The experiment of the optimization run.

        Returns:
        - Union[Dict[str, Union[int, float]], None]: The best parameters for the specified ID if found, otherwise None.

        Raises:
        - ValueError: If the objective function name is not recognized.
        """

        return self.comp_opt.results_.best_para(experiment.objective_function)

    def best_score(self, experiment):
        """
        Return the best score for a specific ID from the results.

        Parameters:
        - experiment (int): The experiment of the optimization run.
        """

        return self.comp_opt.results_.best_score(experiment.objective_function)

    def search_data(self, experiment, times=False):
        """
        Retrieve search data for a specific ID from the results. Optionally exclude evaluation and iteration times if 'times' is set to False.

        Parameters:
        - experiment (int): The experiment of the optimization run.
        - times (bool, optional): Whether to exclude evaluation and iteration times. Defaults to False.

        Returns:
        - pd.DataFrame: The search data for the specified ID.
        """

        search_data_ = self.comp_opt.results_.search_data(experiment.objective_function)

        if times == False:
            search_data_.drop(
                labels=["eval_times", "iter_times"],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return search_data_
