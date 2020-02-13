# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


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

optimizer_dict = {
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


class HyperactiveCore:
    def __init__(self, _main_args_):
        self._main_args_ = _main_args_
        self._opt_args_ = Arguments(**self._main_args_.opt_para)

    def run(self):
        optimizer_class = optimizer_dict[self._main_args_.optimizer]

        dist = Distribution()
        dist.dist(optimizer_class, self._main_args_, self._opt_args_)

        self.results = dist.results
        self.pos_list = dist.pos
        # self.para_list = None
        self.score_list = dist.scores

        self.eval_times = dist.eval_times
        self.iter_times = dist.iter_times
        self.best_scores = dist.best_scores
