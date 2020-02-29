# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


from . import HillClimbingOptimizer, HillClimbingPositioner
from scipy.spatial.distance import euclidean


def gaussian(distance, sig, sigma_factor=1):
    return (
        sigma_factor
        * sig
        * np.exp(-np.power(distance, 2.0) / (sigma_factor * np.power(sig, 2.0)))
    )


class TabuOptimizer(HillClimbingOptimizer):
    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)

    def _tabu_pos(self, pos, _p_):
        _p_.add_tabu(pos)

        return _p_

    def _iterate(self, i, _cand_):
        self._hill_climb_iter(i, _cand_)

        if self._p_.score_new < _cand_.score_best:
            self._p_ = self._tabu_pos(self._p_.pos_new, self._p_)

        return _cand_

    def _init_iteration(self, _cand_):
        self._p_ = super()._init_base_positioner(_cand_, positioner=TabuPositioner)

        self._optimizer_eval(_cand_, self._p_)
        self._update_pos(_cand_, self._p_)

    def _finish_search(self):
        self._pbar_.close_p_bar()

        return self._p_


class TabuPositioner(HillClimbingPositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabus = []
        self.tabu_memory = kwargs["tabu_memory"]

    def add_tabu(self, tabu):
        self.tabus.append(tabu)

        if len(self.tabus) > self.tabu_memory:
            self.tabus.pop(0)

    def move_climb(self, _cand_, pos, epsilon_mod=1):
        sigma = 1 + _cand_._space_.dim * self.epsilon * epsilon_mod
        pos_normal = np.random.normal(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        sigma_mod = 1
        run = True
        while run:
            pos_normal = np.random.normal(pos, sigma * sigma_mod, pos.shape)
            pos_new_int = np.rint(pos_normal)

            p_discard_sum = []
            for tabu in self.tabus:
                distance = euclidean(pos_new_int, tabu)
                sigma_mean = sigma.mean()
                p_discard = gaussian(distance, sigma_mean)

                p_discard_sum.append(p_discard)

            p_discard = np.array(p_discard_sum).sum()
            rand = random.uniform(0, 1)

            if p_discard < rand:
                run = False

            sigma_mod = sigma_mod * 1.01

        n_zeros = [0] * len(_cand_._space_.dim)
        pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)

        return pos.astype(int)
