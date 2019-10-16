# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
import random
import numpy as np

from . import HillClimbingOptimizer


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class StochasticHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _consider(self, _p_, p_accept):
        rand = random.uniform(0, self._arg_.p_down)

        if p_accept > rand:
            _p_.score_current = _p_.score_new
            _p_.pos_current = _p_.pos_new

    def _score_norm(self, _p_):
        return (
            100
            * (_p_.score_current - _p_.score_new)
            / (_p_.score_current + _p_.score_new)
        )

    def _accept(self, _p_):
        return np.exp(-self._score_norm(_p_))

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)
        else:
            p_accept = self._accept(_p_)
            self._consider(_p_, p_accept)

        return _cand_
