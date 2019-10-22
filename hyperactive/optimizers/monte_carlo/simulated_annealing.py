# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = 1

    # use _consider from StochasticHillClimbingOptimizer

    def _accept(self, _p_):
        return np.exp(-self._score_norm(_p_) / self.temp)

    def _iterate(self, i, _cand_, _p_, X, y):
        _cand_, _p_ = self._hill_climb_iteration(_cand_, _p_, X, y)

        if _p_.score_new <= _cand_.score_best:
            p_accept = self._accept(_p_)
            self._consider(_p_, p_accept)

        self.temp = self.temp * self._arg_.annealing_rate

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        return super()._init_base_positioner(_cand_)
