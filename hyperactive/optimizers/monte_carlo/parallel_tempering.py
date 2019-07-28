# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer
from ...base_positioner import BasePositioner


class ParallelTemperingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_para = {"eps": self._arg_.eps}
        self.n_iter_swap = int(self._config_.n_iter / self._arg_.n_swaps)

    def _init_annealers(self, _cand_):
        _p_list_ = []

        for temp in self._arg_.system_temps:
            pos_para = self.pos_para
            pos_para["temp"] = temp

            _p_ = super()._init_base_positioner(
                _cand_, positioner=System, pos_para=pos_para
            )
            _p_list_.append(_p_)

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
        score_diff_norm = (_p1_.score_current - _p2_.score_current) / (
            _p1_.score_current + _p2_.score_current
        )
        temp = (1 / _p1_.temp) - (1 / _p2_.temp)
        return np.exp(score_diff_norm * temp)

    def _annealing_systems(self, _cand_, _p_list_, X, y):
        for _p_ in _p_list_:
            _cand_ = super()._iterate(0, _cand_, _p_, X, y)

        return _cand_

    def _iterate(self, i, _cand_, _p_list_, X, y):
        _cand_ = self._annealing_systems(_cand_, _p_list_, X, y)

        if self.n_iter_swap != 0 and i % self.n_iter_swap == 0:
            self._swap_pos(_cand_, _p_list_)

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        _p_list_ = self._init_annealers(_cand_)

        return _p_list_


class System(BasePositioner):
    def __init__(self, eps=1, temp=1):
        super().__init__(eps)
        self.temp = temp
