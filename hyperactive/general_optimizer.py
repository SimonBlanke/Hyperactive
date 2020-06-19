# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .distribution import Distribution

from .search import Search
from .search_process import SearchProcess

from .verbosity import set_verbosity

from gradient_free_optimizers import (
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
    TreeStructuredParzenEstimators,
    DecisionTreeOptimizer,
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
    "TPE": TreeStructuredParzenEstimators,
    "DecisionTree": DecisionTreeOptimizer,
}


class Optimizer:
    def __init__(
        self,
        obj_func_para=None,
        memory=True,
        max_time=1,
        random_state=1,
        verbosity=3,
        warnings=False,
    ):
        self.study_para = {
            "obj_func_para": obj_func_para,
            "memory": memory,
            "max_time": max_time,
            "random_state": random_state,
            "verbosity": verbosity,
            "warnings": warnings,
        }

        self.search_processes = []

    def add_search(
        self,
        obj_func,
        search_space,
        optimizer="RandomSearch",
        n_iter=10,
        n_jobs=1,
        init=None,
        distribution=None,
    ):
        self.n_jobs = n_jobs

        _info_, _pbar_ = set_verbosity(self.study_para["verbosity"])
        _pbar_ = _pbar_()

        opt_class = optimizer_dict[optimizer]

        for nth_process in range(n_jobs):
            new_search_process = SearchProcess(
                nth_process,
                self.study_para,
                obj_func,
                search_space,
                opt_class,
                n_iter,
                n_jobs,
                init,
                distribution,
                _pbar_,
                _info_,
            )
            self.search_processes.append(new_search_process)

    def run(self):
        search = Search(self.search_processes, self.n_jobs)
        search.run()

        """
        dist = Distribution()
        dist.dist(Search, self._main_args_)

        self.results = dist.results
        self.pos_list = dist.pos
        # self.para_list = None
        self.score_list = dist.scores

        self.eval_times = dist.eval_times
        self.iter_times = dist.iter_times
        self.best_scores = dist.best_scores
        """
