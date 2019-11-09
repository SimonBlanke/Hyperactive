# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

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
    def __init__(self, verbosity, random_state, memory):

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

        self.verbosity = verbosity
        self.random_state = random_state
        self.memory = memory

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

    def search(self, *args, **kwargs):
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

        _core_ = Core(*args, **kwargs)
        _arg_ = Arguments(**_core_.opt_para)

        optimizer_class = self.optimizer_dict[_core_.optimizer]
        self._optimizer_ = optimizer_class(_core_, _arg_)

        self._optimizer_._fit()

        self.pos_list = self._optimizer_.pos_list
        self.score_list = self._optimizer_.score_list

        self.results_params = self._optimizer_.results_params
        self.results_models = self._optimizer_.results_models

        self.total_time = time.time() - start_time

    def get_total_time(self):
        return self.total_time

    def get_eval_time(self):
        return self._optimizer_.eval_time

    def save_report(self):
        pass
