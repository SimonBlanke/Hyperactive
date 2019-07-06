# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

# from .base import BaseOptimizer
from ..local.hill_climbing_optimizer import HillClimber
from .simulated_annealing import SimulatedAnnealingOptimizer


class ParallelTemperingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        memory=True,
        hyperband_init=False,
        eps=1,
        t_rate=0.98,
        n_neighbours=1,
        n_systems=4,
        n_swaps=10,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
            memory,
            hyperband_init,
        )

        self.eps = eps
        self.t_rate = t_rate
        self.temp = 0.1
        self.n_neighbours = n_neighbours
        self.n_systems = n_systems
        self.n_swaps = n_swaps

    def _init_annealers(self, cand):
        ind_list = [Annealer() for _ in range(self.individuals)]
        for ind in ind_list:
            ind.pos = cand._space_.get_random_pos()
            ind.pos_best = ind.pos

        return ind_list

    def _annealing_systems(self, _cand_, ann_list):
        for annealer in ann_list:
            annealer.pos_current = self._annealing(_cand_)

    def _swap_pos(self, _cand_):
        rand = random.uniform(0, 1)

        score_diff_norm = (self.score_current - _cand_.score) / (
            self.score_current + _cand_.score
        )
        temp = (1 / temp1) - (1 / temp2)
        bla = np.exp(score_diff_norm * temp)

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)
        _annealer_ = Annealer()

        n_iter_swap = int(self.n_steps / self.n_swaps)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        self.score_current = _cand_.score

        ann_list = self._init_individuals(_cand_)
        for i in tqdm.tqdm(**self._tqdm_dict(_cand_)):

            _annealer_.find_neighbour(_cand_)
            _cand_.eval(X, y)

            self._annealing_systems(_cand_, ann_list)

            if i % n_iter_swap == 0:
                self._swap_pos(_cand_)

        return _cand_


class Annealer(HillClimber):
    def __init__(self, eps=1):
        super().__init__(eps)

        self.pos = None
        self.pos_best = None

        self.score = -1000
        self.score_best = -1000

    def find_neighbour(self, _cand_):
        super().climb(_cand_)
