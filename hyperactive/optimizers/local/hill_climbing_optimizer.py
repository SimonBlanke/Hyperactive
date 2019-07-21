# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base import BaseOptimizer
from ...base import BasePositioner


class HillClimbingOptimizer(BaseOptimizer):
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
        self.initializer = self._init_climber

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_.pos_best = _p_.pos_new
            _cand_.score_best = _p_.score_new

            _p_.pos_current = _p_.pos_new
            _p_.score_current = _p_.score_new

        return _cand_

    def _init_climber(self, _cand_, X, y):
        _p_ = HillClimber()

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_


class HillClimber(BasePositioner):
    def __init__(self, eps=1):
        super().__init__(eps)
