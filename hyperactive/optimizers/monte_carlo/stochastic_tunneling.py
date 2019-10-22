# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer


class StochasticTunnelingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # _consider same as simulated_annealing

    def _accept(self, _p_):
        f_stun = 1 - np.exp(-self._arg_.gamma * self._score_norm(_p_))
        return np.exp(-f_stun / self.temp)

    # _iterate same as simulated_annealing

    def _init_opt_positioner(self, _cand_, X, y):
        return super()._init_base_positioner(_cand_)
