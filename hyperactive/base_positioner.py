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

        self._pos_best = None
        self._score_best = -np.inf

        self.pos_best_list = []
        self.score_best_list = []

    @property
    def pos_best(self):
        return self._pos_best

    @pos_best.setter
    def pos_best(self, value):
        self.pos_best_list.append(value)
        self._pos_best = value

    @property
    def score_best(self):
        return self._score_best

    @score_best.setter
    def score_best(self, value):
        self.score_best_list.append(value)
        self._score_best = value

    def move_climb(self, _cand_, pos, epsilon_mod=1):
        sigma = 3 + _cand_._space_.dim * self.epsilon * epsilon_mod
        pos_normal = self.climb_dist(pos, sigma, pos.shape)
        pos_new_int = np.rint(pos_normal)

        n_zeros = [0] * len(_cand_._space_.dim)
        pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)

        return pos.astype(int)

    def move_random(self, _cand_):
        return _cand_._space_.get_random_pos()
