# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
import random

from .base import BaseOptimizer


class RandomSearch_Optimizer(BaseOptimizer):
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

    def _move(self, cand):
        cand._space_.get_random_position()

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        for i in tqdm.tqdm(
            range(self.n_steps),
            # desc=str(self.model_str),
            position=_cand_.nth_process,
            leave=False,
        ):

            self._move(_cand_)
            _cand_.eval(X, y)

            if _cand_.score > _cand_.score_best:
                _cand_.score_best = _cand_.score
                _cand_.pos_best = _cand_.pos

        start_point = _cand_._get_warm_start()

        return _cand_.pos_best, _cand_.score_best, start_point
