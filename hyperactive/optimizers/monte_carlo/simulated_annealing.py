# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_para = {"eps": self._arg_.eps}
        self.temp = 0.1

    # use _consider from StochasticHillClimbingOptimizer

    def _accept(self, _p_):
        score_diff_norm = (_p_.score_new - _p_.score_current) / (
            _p_.score_new + _p_.score_current
        )
        return np.exp(-(score_diff_norm / self.temp))

    def _iterate(self, i, _cand_, _p_, X, y):
        _cand_, _p_ = self._hill_climb_iteration(_cand_, _p_, X, y)

        if _p_.score_new <= _cand_.score_best:
            p_accept = self._accept(_p_)
            self._consider(_p_, p_accept)

        self.temp = self.temp * self._arg_.t_rate

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        return super()._init_base_positioner(_cand_, pos_para=self.pos_para)
