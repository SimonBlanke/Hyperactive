# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ...base import BaseOptimizer


class TabuOptimizer(BaseOptimizer):
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
        tabu_memory=[3, 6, 9],
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
        self.tabu_memory = tabu_memory

        self.initializer = self._init_tabu

    def _iterate(self, i, _cand_, X, y):
        self._climber_.climb_tabu(_cand_)
        _cand_.eval(X, y)

        if _cand_.score > _cand_.score_best:
            _cand_.score_best = _cand_.score
            _cand_.pos_best = _cand_.pos

            self._climber_.tabu_memory_short.append(_cand_.pos_best)

            if len(self._climber_.tabu_memory_short) > self.tabu_memory[0]:
                del self._climber_.tabu_memory_short[0]

        return _cand_

    def _init_tabu(self, _cand_):
        self._climber_ = HillClimber(self.eps)


class HillClimber:
    def __init__(self, eps):
        self.eps = eps

        self.tabu_memory_short = []

    def climb_tabu(self, _cand_, eps_mod=1):
        sigma = (_cand_._space_.dim / 100) * self.eps * eps_mod

        in_tabu_mem = True
        while in_tabu_mem:
            pos_new = np.random.normal(_cand_.pos_best, sigma, _cand_.pos.shape)
            pos_new_int = np.rint(pos_new)

            if not any((pos_new_int == pos).all() for pos in self.tabu_memory_short):
                in_tabu_mem = False

        n_zeros = [0] * len(_cand_._space_.dim)
        _cand_.pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)
