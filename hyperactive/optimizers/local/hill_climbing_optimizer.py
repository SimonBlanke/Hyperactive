# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
import numpy as np

from ...base import BaseOptimizer


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
        hyperband_init=False,
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
            hyperband_init,
        )

        self.eps = eps

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)
        _climber_ = HillClimber(self.eps)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        for i in tqdm.tqdm(**self._tqdm_dict(_cand_)):

            _climber_.climb(_cand_)
            _cand_.eval(X, y)

            if _cand_.score > _cand_.score_best:
                _cand_.score_best = _cand_.score
                _cand_.pos_best = _cand_.pos

        return _cand_


class HillClimber:
    def __init__(self, eps):
        self.eps = eps

    def climb(self, _cand_, eps_mod=1):
        sigma = (_cand_._space_.dim / 100) * self.eps * eps_mod
        pos_new = np.random.normal(_cand_.pos_best, sigma, _cand_.pos.shape)
        pos_new_int = np.rint(pos_new)

        n_zeros = [0] * len(_cand_._space_.dim)
        _cand_.pos = np.clip(pos_new_int, n_zeros, _cand_._space_.dim)
