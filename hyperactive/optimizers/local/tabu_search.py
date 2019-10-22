# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


from . import HillClimbingOptimizer
from ...base_positioner import BasePositioner
from scipy.spatial.distance import euclidean


def gaussian(distance, sig):
    return sig * np.exp(-np.power(distance, 2.0) / (2 * np.power(sig, 2.0)))


class TabuOptimizer(HillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tabu_pos(self, pos, _p_):
        _p_.add_tabu(pos)

        return _p_

    def _iterate(self, i, _cand_, _p_, X, y):
        _cand_, _p_ = self._hill_climb_iter(_cand_, _p_, X, y)

        if _p_.score_new <= _cand_.score_best:
            _p_ = self._tabu_pos(_p_.pos_new, _p_)

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        return super()._init_base_positioner(_cand_, positioner=TabuPositioner)


class TabuPositioner(BasePositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabus = []

    def add_tabu(self, tabu):
        self.tabus.append(tabu)

    def move_climb(self, _cand_, pos, epsilon_mod=1):
        sigma = 3 + _cand_._space_.dim * self.epsilon * epsilon_mod
        pos_normal = np.random.normal(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        for tabu in self.tabus:
            distance = euclidean(pos_new_int, tabu)
            sigma_mean = sigma.mean()
            p_discard = gaussian(distance, sigma_mean)
            rand = random.uniform(0, 1)

            if p_discard > rand:
                pos_normal = np.random.normal(pos, sigma, pos.shape)
                pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(_cand_._space_.dim)
        pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)

        return pos
