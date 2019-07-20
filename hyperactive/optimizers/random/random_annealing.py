# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base import BaseOptimizer
from ..local.hill_climbing_optimizer import HillClimber


class RandomAnnealingOptimizer(BaseOptimizer):
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
        eps=100,
        t_rate=0.98,
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
        self.temp = 10

        self.initializer = self._init_rnd_annealing

    def _iterate(self, i, _cand_, X, y):
        self.temp = self.temp * self.t_rate

        self._annealer_.find_neighbour(_cand_, self.temp)
        _cand_.pos = self._annealer_.pos
        _cand_.eval(X, y)

        if _cand_.score > _cand_.score_best:
            _cand_.score_best = _cand_.score
            _cand_.pos_best = _cand_.pos

        return _cand_

    def _init_rnd_annealing(self, _cand_):
        self._annealer_ = Annealer(self.eps)

        self.score_current = _cand_.score


class Annealer(HillClimber):
    def __init__(self, eps):
        super().__init__(eps)

    def find_neighbour(self, _cand_, eps_mod):
        super().climb(_cand_)
