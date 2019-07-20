# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base import BaseOptimizer
from ...base import BasePositioner


class RandomRestartHillClimbingOptimizer(BaseOptimizer):
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
        n_restarts=10,
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
        self.n_restarts = n_restarts

        self.n_iter_restart = int(self.n_iter / self.n_restarts)

        self.initializer = self._init_rr_climber

    def _iterate(self, i, _cand_, X, y):
        self._climber_.climb(_cand_)
        _cand_.pos = self._climber_.pos
        _cand_.eval(X, y)

        if _cand_.score > _cand_.score_best:
            _cand_.score_best = _cand_.score
            _cand_.pos_best = _cand_.pos

        if self.n_iter_restart != 0 and i % self.n_iter_restart == 0:
            _cand_.pos = _cand_._space_.get_random_pos()

        return _cand_

    def _init_rr_climber(self, _cand_):
        self._climber_ = HillClimber(self.eps)

        _cand_.pos_current = _cand_.pos
        _cand_.score_current = _cand_.score


class HillClimber(BasePositioner):
    def __init__(self, eps):
        self.eps = eps
