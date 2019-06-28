# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer


class SimulatedAnnealing_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        eps=1,
        t_rate=0.99,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
        )

        self.eps = eps
        self.t_rate = t_rate
        self.temp = 0.1

    def _move(self, cand):
        pos = {}

        for pos_key in cand.pos:
            n_values = len(cand._space_.para_space[pos_key])
            rand_eps = random.randint(-self.eps, self.eps + 1)

            index = cand.pos[pos_key]
            index_new = index + rand_eps
            pos[pos_key] = index_new

            # don't go out of range
            if index_new < 0:
                index_new = 0
            if index_new > n_values - 1:
                index_new = n_values - 1

            pos[pos_key] = index_new

        cand.pos = pos

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        self.score_current = _cand_.score

        for i in tqdm.tqdm(
            range(self.n_steps),
            # desc=str(self.model_str),
            position=nth_process,
            leave=False,
        ):
            self.temp = self.temp * self.t_rate
            rand = random.randint(0, 1)

            self._move(_cand_)
            _cand_.eval(X, y)

            # Normalized score difference to have a factor for later use with temperature and random
            score_diff_norm = (self.score_current - _cand_.score) / (
                self.score_current + _cand_.score
            )

            if _cand_.score > self.score_current:
                self.score_current = _cand_.score
                self.pos_curr = _cand_.pos

                if _cand_.score > _cand_.score_best:
                    _cand_.score_best = _cand_.score
                    self.pos_curr = _cand_.pos

            elif np.exp(score_diff_norm / self.temp) > rand:
                self.score_current = _cand_.score
                self.hyperpara_indices_current = _cand_.pos

        start_point = _cand_._get_warm_start()

        return _cand_.pos_best, _cand_.score_best, start_point
