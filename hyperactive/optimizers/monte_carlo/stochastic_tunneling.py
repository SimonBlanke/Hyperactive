# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from .simulated_annealing import SimulatedAnnealingOptimizer
from ...base import BasePositioner


class StochasticTunnelingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        memory=True,
        scatter_init=False,
        eps=1,
        t_rate=0.98,
        n_neighbours=1,
        gamma=1,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
            memory,
            scatter_init,
        )

        self.eps = eps
        self.t_rate = t_rate
        self.n_neighbours = n_neighbours
        self.gamma = gamma
        self.temp = 0.01

        self.initializer = self._init_tunneling

    # _consider same as simulated_annealing

    def _accept(self, _p_):
        score_diff_norm = (_p_.score_new - _p_.score_current) / (
            _p_.score_new + _p_.score_current
        )
        f_stun = 1 - np.exp(-self.gamma * score_diff_norm)
        return np.exp(-f_stun / self.temp)

    # _iterate same as simulated_annealing

    def _init_tunneling(self, _cand_, X, y):
        _p_ = HillClimber()

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_


class HillClimber(BasePositioner):
    def __init__(self, eps=1):
        super().__init__(eps)
