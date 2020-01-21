# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .main_args import MainArgs
from .opt_args import Arguments
from .distribution import Distribution

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


class Hyperactive:
    def __init__(
        self, X, y, memory="long", random_state=False, verbosity=3, warnings=False
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
        self._main_args_.search_args(
            search_config, max_time, n_iter, optimizer, n_jobs, init_config
        )
        self._opt_args_ = Arguments(**self._main_args_.opt_para)
        optimizer_class = self.optimizer_dict[self._main_args_.optimizer]

        dist = Distribution()
        dist.dist(optimizer_class, self._main_args_, self._opt_args_)

        self.results = dist.results
        self.pos = dist.pos
        self.para_list = None
        self.scores = dist.scores

        self.eval_times = dist.eval_times
        self.opt_times = dist.opt_times
