# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


class BasePositioner:
    def __init__(self, epsilon=0.03):
        self.epsilon = epsilon

        self.pos_new = None
        self.score_new = -1000

        self.pos_current = None
        self.score_current = -1000

        self.pos_best = None
        self.score_best = -1000

    def move_climb(self, _cand_, pos, epsilon_mod=1):
        sigma = 3 + _cand_._space_.dim * self.epsilon * epsilon_mod
        pos_normal = np.random.normal(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(_cand_._space_.dim)
        pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)

        return pos

    def move_random(self, _cand_):
        return _cand_._space_.get_random_pos()
