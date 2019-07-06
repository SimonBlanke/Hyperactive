# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm

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
        hyperband_init=False,
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
            hyperband_init,
        )

        self.eps = eps
        self.t_rate = t_rate
        self.temp = 0.1

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)
        _annealer_ = Annealer(self.eps)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        self.score_current = _cand_.score

        for i in tqdm.tqdm(**self._tqdm_dict(_cand_)):
            self.temp = self.temp * self.t_rate

            _annealer_.find_neighbour(_cand_, self.temp)
            _cand_.eval(X, y)

            if _cand_.score > _cand_.score_best:
                _cand_.score_best = _cand_.score
                _cand_.pos_best = _cand_.pos

        return _cand_


class Annealer(HillClimber):
    def __init__(self, eps):
        super().__init__(eps)

    def find_neighbour(self, _cand_, eps_mod):
        super().climb(_cand_)
