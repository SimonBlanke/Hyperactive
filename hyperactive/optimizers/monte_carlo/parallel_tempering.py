# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer
from ...base import BasePositioner


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
        scatter_init=False,
        eps=1,
        t_rate=0.98,
        n_neighbours=1,
        system_temps=[0.1, 0.2, 0.01],
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
            scatter_init,
        )

        self.eps = eps
        self.t_rate = t_rate
        self.temp = 0.1
        self.n_neighbours = n_neighbours
        self.system_temps = system_temps
        self.n_swaps = n_swaps

        self.n_iter_swap = int(self.n_iter / self.n_swaps)

        self.initializer = self._init_tempering

    def _init_annealers(self, _cand_):
        _p_list_ = [HillClimber(temp=temp) for temp in self.system_temps]
        for _p_ in _p_list_:
            _p_.pos_current = _p_.move_random(_cand_)
            _p_.pos_best = _p_.pos_current

        return _p_list_

    def _swap_pos(self, _cand_, _p_list_):
        for _p1_ in _p_list_:
            for _p2_ in _p_list_:
                rand = random.uniform(0, 1)

                p_accept = self._accept_swap(_p1_, _p2_)
                # print("p_accept", p_accept)

                if p_accept > rand:
                    temp_temp = _p1_.temp  # haha!
                    _p1_.temp = _p2_.temp
                    _p1_.temp = temp_temp
                    break

    def _accept_swap(self, _p1_, _p2_):
        score_diff_norm = (_p1_.score_current - _p2_.score_current) / (
            _p1_.score_current + _p2_.score_current
        )
        temp = (1 / _p1_.temp) - (1 / _p2_.temp)
        return np.exp(score_diff_norm * temp)

    def _annealing_systems(self, _cand_, _p_list_, X, y):
        for _p_ in _p_list_:
            _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)
            _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

            if _p_.score_new > _cand_.score_best:
                _cand_.score_best = _p_.score_new
                _cand_.pos_best = _p_.pos_new

                _p_.pos_current = _p_.pos_new
                _p_.score_current = _p_.score_new
            else:
                p_accept = self._accept(_p_)
                self._consider(_p_, p_accept)

            _p_.temp = _p_.temp * self.t_rate

    def _iterate(self, i, _cand_, _p_list_, X, y):
        self._annealing_systems(_cand_, _p_list_, X, y)

        if self.n_iter_swap != 0 and i % self.n_iter_swap == 0:
            self._swap_pos(_cand_, _p_list_)

        return _cand_

    def _init_tempering(self, _cand_, X, y):
        _p_list_ = self._init_annealers(_cand_)

        return _p_list_


class HillClimber(BasePositioner):
    def __init__(self, eps=1, temp=1):
        super().__init__(eps)
        self.temp = temp
