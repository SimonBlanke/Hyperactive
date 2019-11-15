# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from . import HillClimbingOptimizer


class StochasticHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)

    def _consider(self, _p_, p_accept):
        rand = random.uniform(0, self._opt_args_.p_down)

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

    def _stochastic_hill_climb_iter(self, _cand_, _p_):
        _cand_, _p_ = self._hill_climb_iter(_cand_, _p_)

        if _p_.score_new <= _cand_.score_best:
            p_accept = self._accept(_p_)
            self._consider(_p_, p_accept)

        return _cand_

    def _iterate(self, i, _cand_, _p_):
        _cand_ = self._stochastic_hill_climb_iter(_cand_, _p_)

        return _cand_
