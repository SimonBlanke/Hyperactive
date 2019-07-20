# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np

from ...base import BaseOptimizer


class StochasticHillClimbingOptimizer(BaseOptimizer):
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
        r=1e-6,
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
        self.r = r

        self.initializer = self._init_stoch_climber

    def _iterate(self, i, _cand_, X, y):
        rand = random.uniform(0, self.r)

        self._climber_.climb(_cand_)
        _cand_.pos = self._climber_.pos
        _cand_.eval(X, y)

        score_diff_norm = abs(_cand_.score_current - _cand_.score) / (
            _cand_.score_current + _cand_.score
        )

        if _cand_.score > _cand_.score_best:
            _cand_.score_best = _cand_.score
            _cand_.pos_best = _cand_.pos

        elif score_diff_norm > rand:
            _cand_.pos_current = _cand_.pos
            _cand_.score_current = _cand_.score

        return _cand_

    def _init_stoch_climber(self, _cand_):
        self._climber_ = HillClimber(self.eps)

        _cand_.pos_current = _cand_.pos
        _cand_.score_current = _cand_.score


class HillClimber:
    def __init__(self, eps):
        self.eps = eps

    def climb(self, _cand_, eps_mod=1):
        sigma = (_cand_._space_.dim / 33) * self.eps / eps_mod
        pos_new = np.random.normal(_cand_.pos_current, sigma, _cand_.pos.shape)
        pos_new_int = np.rint(pos_new)

        n_zeros = [0] * len(_cand_._space_.dim)
        self.pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)
