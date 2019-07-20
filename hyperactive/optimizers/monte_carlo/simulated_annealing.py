# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from ...base import BaseOptimizer
from ...base import BasePositioner


class SimulatedAnnealingOptimizer(BaseOptimizer):
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
        self.temp = 0.1

        self.initializer = self._init_annealing

    def _annealing(self, _cand_):
        self.temp = self.temp * self.t_rate
        rand = random.uniform(0, 1)

        # Normalized score difference to have a factor for later use with temperature and random
        score_diff_norm = (self._annealer_.score - _cand_.score) / (
            self._annealer_.score + _cand_.score
        )
        p_accept = np.exp(-(score_diff_norm / self.temp))

        if _cand_.score > self._annealer_.score:
            self._annealer_.score = _cand_.score
            self._annealer_.pos = _cand_.pos

            if _cand_.score > _cand_.score_best:
                _cand_.score_best = _cand_.score
                self._annealer_.pos = _cand_.pos

        elif p_accept > rand:
            self._annealer_.score = _cand_.score
            self._annealer_.pos = _cand_.pos

        return self._annealer_.pos

    def _iterate(self, i, _cand_, X, y):
        self._annealer_.climb(_cand_)
        _cand_.pos = self._annealer_.pos
        _cand_.eval(X, y)

        # print("_cand_.pos", _cand_.pos)

        self._annealing(_cand_)

        return _cand_

    def _init_annealing(self, _cand_):
        self._annealer_ = Annealer()

        self._annealer_.pos = _cand_.pos
        self._annealer_.score = _cand_.score


class Annealer(BasePositioner):
    def __init__(self, eps=1):
        super().__init__(eps)
