# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initializer = self._init_random

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_random(_cand_)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_.score_best = _p_.score_new
            _cand_.pos_best = _p_.pos_new

        return _cand_

    def _init_random(self, _cand_, X, y):
        return super()._initialize(_cand_)
