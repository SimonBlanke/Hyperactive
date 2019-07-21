# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

from ...base import BaseOptimizer
from ...base import BasePositioner


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

    def _consider(self, _p_, p_accept):
        rand = random.uniform(0, 1)

        if p_accept > rand:
            _p_.score_current = _p_.score_new
            _p_.pos_current = _p_.pos_new

    def _accept(self, _p_):
        return (_p_.score_new - _p_.score_current) / (_p_.score_new + _p_.score_current)

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_.score_best = _p_.score_new
            _cand_.pos_best = _p_.pos_new

            _p_.pos_current = _p_.pos_new
            _p_.score_current = _p_.score_new
        else:
            p_accept = self._accept(_p_)
            self._consider(_p_, p_accept)

        return _cand_

    def _init_stoch_climber(self, _cand_, X, y):
        _p_ = HillClimber()

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_


class HillClimber(BasePositioner):
    def __init__(self, eps=1):
        self.eps = eps
