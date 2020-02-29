# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer


class StochasticTunnelingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)

    def _accept(self, _p_):
        f_stun = 1 - np.exp(-self._opt_args_.gamma * self._score_norm(_p_))
        return np.exp(-f_stun / self.temp)
