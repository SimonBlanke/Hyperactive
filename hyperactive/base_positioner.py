# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class BasePositioner:
    def __init__(self, *args, **kwargs):
        self.pos_new = None
        self.score_new = -np.inf

        self.pos_current = None
        self.score_current = -np.inf

        self.pos_best = None
        self.score_best = -np.inf

    def move_climb(self, _cand_, pos, epsilon_mod=1):
        sigma = 3 + _cand_._space_.dim * self.epsilon * epsilon_mod
        pos_normal = self.climb_dist(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(_cand_._space_.dim)
        pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)

        return pos

    def move_random(self, _cand_):
        return _cand_._space_.get_random_pos()
