# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer
from ..local import HillClimbingPositioner


class ParallelTemperingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)
        self.n_iter_swap = int(self._main_args_.n_iter / self._opt_args_.n_swaps)

    def _init_annealers(self, _cand_):
        _p_list_ = [
            System(**self._opt_args_.kwargs_opt, temp=temp)
            for temp in self._opt_args_.system_temperatures
        ]

        for _p_ in _p_list_:
            _p_.pos_new = _cand_._space_.get_random_pos()

            self._optimizer_eval(_cand_, _p_)
            _p_.pos_current = _p_.pos_new
            _p_.score_current = _p_.score_new

        return _p_list_

    def _swap_pos(self, _cand_, _p_list_):
        _p_list_temp = _p_list_[:]

        for _p1_ in _p_list_:
            rand = random.uniform(0, 1)
            _p2_ = np.random.choice(_p_list_temp)

            p_accept = self._accept_swap(_p1_, _p2_)
            if p_accept > rand:
                temp_temp = _p1_.temp  # haha!
                _p1_.temp = _p2_.temp
                _p2_.temp = temp_temp

    def _accept_swap(self, _p1_, _p2_):
        denom = _p1_.score_current + _p2_.score_current

        if denom == 0:
            return 100
        elif abs(denom) == np.inf:
            return 0
        else:
            score_diff_norm = (_p1_.score_current - _p2_.score_current) / denom

            temp = (1 / _p1_.temp) - (1 / _p2_.temp)
            return np.exp(score_diff_norm * temp)

    def _anneal_system(self, _cand_, _p_):
        self._p_ = _p_
        super()._iterate(0, _cand_)

    def _iterate(self, i, _cand_):
        _p_current = self._p_list_[i % len(self._p_list_)]

        self._anneal_system(_cand_, _p_current)

        if self.n_iter_swap != 0 and i % self.n_iter_swap == 0:
            self._swap_pos(_cand_, self._p_list_)

        return _cand_

    def _init_iteration(self, _cand_):
        self._p_list_ = self._init_annealers(_cand_)

    def _finish_search(self):
        self._pbar_.close_p_bar()

        return self._p_list_


class System(HillClimbingPositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = kwargs["temp"]
