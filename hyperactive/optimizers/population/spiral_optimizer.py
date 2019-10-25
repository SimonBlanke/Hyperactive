# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from . import ParticleSwarmOptimizer
from ...base_positioner import BasePositioner


class SpiralOptimizer(ParticleSwarmOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 0.001

    def _rotation_matrix(self, _cand_):
        n_dim = len(_cand_._space_.dim)
        zero = np.zeros((n_dim, n_dim))
        I = np.identity(n_dim)

        R = np.add(zero, I)
        R[0, n_dim] = -1

        return R

    def _update_r(self, i_max):
        return np.power(self.delta, i_max)

    def _center(self, _p_list_):
        pos_list = []

        for _p_ in _p_list_:
            pos_list.append(_p_.pos_current)

        pos_list = np.array(pos_list)
        pos_mean = pos_list.mean()
        pos_mean_int = np.rint(pos_mean)

        return pos_mean

    def _move_positioners(self, _cand_, _p_list_):
        for _p_ in _p_list_:

            pos_new = pos_mean + r * R * (_p_.pos_current - pos_mean) 

            _p_.pos_new = pos_new
