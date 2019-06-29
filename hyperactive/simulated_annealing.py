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
        pos = self._get_neighbour(cand.pos)

        # limit movement
        n_zeros = [0] * len(cand._space_.n_values)
        pos = np.clip(pos, n_zeros, cand._space_.n_values)

        cand.pos = pos

    def _get_neighbour(self, pos_np):
        pos_np_new = []

        rand_eps = np.random.randint(-self.eps, self.eps + 1, size=pos_np.shape)
        pos_np_new = pos_np + rand_eps

        return pos_np_new

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        self.score_current = _cand_.score

        for i in tqdm.tqdm(**self._tqdm_dict(_cand_)):
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

        return _cand_
