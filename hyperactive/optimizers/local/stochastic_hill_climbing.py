# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from . import HillClimbingOptimizer


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

    def _stochastic_hill_climb_iter(self, _cand_, _p_, X, y):
        _cand_, _p_ = self._hill_climb_iter(_cand_, _p_, X, y)

        if _p_.score_new <= _cand_.score_best:
            p_accept = self._accept(_p_)
            self._consider(_p_, p_accept)

        return _cand_

    def _iterate(self, i, _cand_, _p_, X, y):
        _cand_ = self._stochastic_hill_climb_iter(_cand_, _p_, X, y)

        return _cand_
