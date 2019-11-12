# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import ray

from .core import Core
from .opt_args import Arguments
from . import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    TabuOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticTunnelingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
)


class Hyperactive:
    def __init__(self, *args, **kwargs):

        """

        Parameters
        ----------

        search_config: dict
            A dictionary providing the model and hyperparameter search space for the
            optimization process.
        n_iter: int
            The number of iterations the optimizer performs.
        metric: string, optional (default: "accuracy")
            The metric the model is evaluated by.
        n_jobs: int, optional (default: 1)
            The number of searches to run in parallel.
        cv: int, optional (default: 3)
            The number of folds for the cross validation.
        verbosity: int, optional (default: 1)
            Verbosity level. 1 prints out warm_start points and their scores.
        random_state: int, optional (default: None)
            Sets the random seed.
        warm_start: dict, optional (default: False)
            Dictionary that definies a start point for the optimizer.
        memory: bool, optional (default: True)
            A memory, that saves the evaluation during the optimization to save time when
            optimizer returns to position.
        scatter_init: int, optional (default: False)
            Defines the number n of random positions that should be evaluated with 1/n the
            training data, to find a better initial position.

        Returns
        -------
        None

        """

        self._core_ = Core(*args, **kwargs)
        self._arg_ = Arguments(**self._core_.opt_para)

        self.optimizer_dict = {
            "HillClimbing": HillClimbingOptimizer,
            "StochasticHillClimbing": StochasticHillClimbingOptimizer,
            "TabuSearch": TabuOptimizer,
            "RandomSearch": RandomSearchOptimizer,
            "RandomRestartHillClimbing": RandomRestartHillClimbingOptimizer,
            "RandomAnnealing": RandomAnnealingOptimizer,
            "SimulatedAnnealing": SimulatedAnnealingOptimizer,
            "StochasticTunneling": StochasticTunnelingOptimizer,
            "ParallelTempering": ParallelTemperingOptimizer,
            "ParticleSwarm": ParticleSwarmOptimizer,
            "EvolutionStrategy": EvolutionStrategyOptimizer,
            "Bayesian": BayesianOptimizer,
        }

    def search(self, search_config, n_iter=100, max_time=None, n_jobs=1):
        """Public method for starting the search with the training data (X, y)

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]

        Returns
        -------
        None
        """
        start_time = time.time()

        self._core_.search_config = search_config
        self._core_.n_iter = n_iter
        self._core_.max_time = max_time
        self._core_.n_jobs = n_jobs
        self._core_.n_models = len(list(self._core_.search_config.keys()))

        optimizer_class = self.optimizer_dict[self._core_.optimizer]

        if ray.is_initialized():
            optimizer_class = ray.remote(optimizer_class)
            opts = [
                optimizer_class.remote(self._core_, self._arg_)
                for i in range(self._core_.n_jobs)
            ]
            jobs = [o._search.remote(i) for i, o in enumerate(opts)]
            _cand_ = ray.get(jobs)

            process = [o._process_results.remote(cand) for o, cand in zip(opts, _cand_)]
            ray.get(process)

        else:
            self._optimizer_ = optimizer_class(self._core_, self._arg_)
            self._optimizer_._search()

        """

        self.pos_list = self._optimizer_.pos_list
        self.score_list = self._optimizer_.score_list

        self.results_params = self._optimizer_.results_params
        self.results_models = self._optimizer_.results_models

        """

        self.total_time = time.time() - start_time

    def get_total_time(self):
        return self.total_time

    def get_eval_time(self):
        return self._optimizer_.eval_time

    def save_report(self):
        pass
