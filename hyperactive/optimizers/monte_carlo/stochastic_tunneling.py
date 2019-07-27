# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer


class StochasticTunnelingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initializer = self._init_tunneling
        self.temp = 0.01

    # _consider same as simulated_annealing

    def _accept(self, _p_):
        score_diff_norm = (_p_.score_new - _p_.score_current) / (
            _p_.score_new + _p_.score_current
        )
        f_stun = 1 - np.exp(-self.gamma * score_diff_norm)
        return np.exp(-f_stun / self.temp)

    # _iterate same as simulated_annealing

    def _init_tunneling(self, _cand_, X, y):
        return super()._initialize(_cand_, pos_para=self.pos_para)
