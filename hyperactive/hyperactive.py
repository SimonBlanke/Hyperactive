# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import warnings

from .main_args import MainArgs
from .opt_args import Arguments

from .optimizers import (
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


def stop_warnings():
    # because sklearn warnings are annoying when they appear 100 times
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn


def try_ray_import():
    try:
        import ray

        if ray.is_initialized():
            rayInit = True
        else:
            rayInit = False
    except ImportError:
        warnings.warn("failed to import ray", ImportWarning)
        ray = None
        rayInit = False

    return ray, rayInit


class Hyperactive:
    def __init__(
        self, X, y, memory="long", random_state=1, verbosity=3, warnings=False
    ):
        self.X = X
        self._main_args_ = MainArgs(X, y, memory, random_state, verbosity)

        if not warnings:
            stop_warnings()

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

    def search(
        self,
        search_config,
        n_iter=10,
        max_time=None,
        optimizer="RandomSearch",
        n_jobs=1,
        init_config=None,
    ):

        start_time = time.time()

        self._main_args_.search_args(
            search_config, max_time, n_iter, optimizer, n_jobs, init_config
        )
        self._opt_args_ = Arguments(self._main_args_.opt_para)
        optimizer_class = self.optimizer_dict[self._main_args_.optimizer]

        ray, rayInit = try_ray_import()

        if rayInit:
            optimizer_class = ray.remote(optimizer_class)
            opts = [
                optimizer_class.remote(self._main_args_, self._opt_args_)
                for job in range(self._main_args_.n_jobs)
            ]
            searches = [
                opt.search.remote(job, rayInit=rayInit) for job, opt in enumerate(opts)
            ]
            self.results_params, self.results_models, self.pos_list, self.score_list, self.eval_time = ray.get(
                searches
            )[
                0
            ]

            ray.shutdown()
        else:
            self._optimizer_ = optimizer_class(self._main_args_, self._opt_args_)
            self.results_params, self.results_models, self.pos_list, self.score_list, self.eval_time = (
                self._optimizer_.search()
            )

        self.total_time = time.time() - start_time
