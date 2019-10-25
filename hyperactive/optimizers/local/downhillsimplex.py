# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


from . import HillClimbingOptimizer
from ...base_positioner import BasePositioner


class DownhillSimplexOptimizer(HillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_simplex(self, _cand_):
        return _p_list

    def _rank_individuals(self, _p_list_):
        scores_list = []
        for _p_ in _p_list_:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        return idx_sorted_ind

    def _get_middle(self, _p_list_):
        pos_list = []

        for _p_ in _p_list_:
            pos_list.append(_p_.pos_current)

        pos_list = np.array(pos_list)
        pos_mean = pos_list.mean()
        pos_mean_int = np.rint(pos_mean)

        return pos_mean

    def _reflect_point(self, _p_, middle):
        return _p_

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_random(_cand_)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_.score_best = _p_.score_new
            _cand_.pos_best = _p_.pos_new

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        return super()._init_base_positioner(_cand_, positioner=DownhillSimplexPositioner)


class DownhillSimplexPositioner(BasePositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def move(self, _cand_):

        return pos
