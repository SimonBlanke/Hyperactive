"""Base class for optimizer."""

from typing import Union, List, Dict
import multiprocessing as mp
import pandas as pd

from .backend_stuff.search_space import SearchSpace
from .hyper_optimizer import HyperOptimizer


from ..composite_optimizer import CompositeOptimizer

from skbase.base import BaseObject


class BaseOptimizer(BaseObject):
    """Base class for optimizer."""

    opt_pros: dict

    def __init__(self, optimizer_class, opt_params):
        super().__init__()
        self.opt_params = opt_params
        self.hyper_optimizer = HyperOptimizer(optimizer_class, opt_params)

        self.opt_pros = {}

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

        self.check_list(search_space)

        constraints = constraints or []
        pass_through = pass_through or {}
        early_stopping = early_stopping or {}

        search_id = self._default_search_id(
            search_id, experiment.objective_function
        )
        s_space = SearchSpace(search_space)
        self.verbosity = verbosity

        self.hyper_optimizer.setup_search(
            experiment=experiment,
            s_space=s_space,
            n_iter=n_iter,
            initialize=initialize,
            constraints=constraints,
            pass_through=pass_through,
            callbacks=experiment.callbacks,
            catch=experiment.catch,
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

    def __add__(self, optimizer_instance):
        return CompositeOptimizer(self, optimizer_instance)

    def run(
        self,
        max_time=None,
        distribution: str = "multiprocessing",
        n_processes: Union[str, int] = "auto",
    ):
        self.comp_opt = CompositeOptimizer(self)
        self.comp_opt.run(max_time, distribution, n_processes, self.verbosity)

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

        return self.comp_opt.results_.best_para(id_)

    def best_score(self, id_):
        """
        Return the best score for a specific ID from the results.

        Parameters:
        - id_ (int): The ID for which the best score is requested.
        """

        return self.comp_opt.results_.best_score(id_)

    def search_data(self, id_, times=False):
        """
        Retrieve search data for a specific ID from the results. Optionally exclude evaluation and iteration times if 'times' is set to False.

        Parameters:
        - id_ (int): The ID of the search data to retrieve.
        - times (bool, optional): Whether to exclude evaluation and iteration times. Defaults to False.

        Returns:
        - pd.DataFrame: The search data for the specified ID.
        """

        search_data_ = self.comp_opt.results_.search_data(
            id_.objective_function
        )

        if times == False:
            search_data_.drop(
                labels=["eval_times", "iter_times"],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return search_data_
