# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numpy as np
import multiprocessing

from .verbosity import set_verbosity

from .search_process import SearchProcess


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


class Search:
    def __init__(self, _main_args_):
        self._main_args_ = _main_args_

        self._info_, _pbar_ = set_verbosity(_main_args_.verbosity)
        self._pbar_ = _pbar_()

    def search(self, nth_process=0, rayInit=False):
        self.start_time = time.time()
        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}
        self.pos_list = {}
        self.score_list = {}

        if rayInit:
            self._run_job(nth_process)
        elif self._main_args_.n_jobs == 1:
            self._run_job(nth_process)
        else:
            self._run_multiple_jobs()

        return (
            self.results,
            self.pos_list,
            self.score_list,
            self.eval_times,
            self.iter_times,
            self.best_scores,
        )

    def _search_multiprocessing(self):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = multiprocessing.Pool(self._main_args_.n_jobs)
        self.processlist, _p_list = zip(
            *pool.map(self._search, self._main_args_._n_process_range)
        )

        return self.processlist, _p_list

    def _run_job(self, nth_process):
        self.process, _p_ = self._search(nth_process)
        self._get_attributes(_p_)

    def _get_attributes(self, _p_):
        self.results[self.process.func_] = self.process._process_results()
        self.eval_times[self.process.func_] = self.process.eval_time
        self.iter_times[self.process.func_] = self.process.iter_times
        self.best_scores[self.process.func_] = self.process.score_best

        if isinstance(_p_, list):
            self.pos_list[self.process.func_] = [np.array(p.pos_list) for p in _p_]
            self.score_list[self.process.func_] = [np.array(p.score_list) for p in _p_]
        else:
            self.pos_list[self.process.func_] = [np.array(_p_.pos_list)]
            self.score_list[self.process.func_] = [np.array(_p_.score_list)]

    def _run_multiple_jobs(self):
        self.processlist, _p_list = self._search_multiprocessing()

        for _ in range(int(self._main_args_.n_jobs / 2) + 2):
            print("\n")

        for self.process, _p_ in zip(self.processlist, _p_list):
            self._get_attributes(_p_)

    def _search(self, nth_process):
        self._initialize_search(self._main_args_, nth_process, self._info_)

        n_positions = 10

        init_positions = self.process.init_pos(n_positions)

        self.opt = optimizer_dict[self._main_args_.optimizer](
            init_positions, self.process._space_.dim, self._main_args_.opt_para
        )

        # loop to initialize N positions
        for nth_init in range(len(init_positions)):
            pos_new = self.opt.init_pos(nth_init)
            score_new = self._get_score(pos_new, 0)
            self.opt.evaluate(score_new)

        # loop to do the iterations
        for nth_iter in range(len(init_positions), self._main_args_.n_iter):
            pos_new = self.opt.iterate(nth_iter)
            score_new = self._get_score(pos_new, nth_iter)
            self.opt.evaluate(score_new)

        self._pbar_.close_p_bar()

        return self.process, self.opt.p_list

    def _get_score(self, pos_new, nth_iter):
        score_new = self.process.eval_pos(pos_new, self._pbar_, nth_iter)
        self._pbar_.update_p_bar(1, self.process)

        if score_new > self.process.score_best:
            self.process.score = score_new
            self.process.pos = pos_new

        return score_new

    def _time_exceeded(self):
        run_time = time.time() - self.start_time
        return self._main_args_.max_time and run_time > self._main_args_.max_time

    def _initialize_search(self, _main_args_, nth_process, _info_):
        _main_args_._set_random_seed(nth_process)

        self.process = SearchProcess(nth_process, _main_args_, _info_)
        self._pbar_.init_p_bar(nth_process, self._main_args_)
