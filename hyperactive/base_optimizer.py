# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .base_positioner import BasePositioner


class BaseOptimizer:
    def __init__(self, _opt_args_):
        self._opt_args_ = _opt_args_

    def _update_pos(self, _cand_, _p_):
        if _p_.score_new > _p_.score_best:
            _p_.pos_best = _p_.pos_new
            _p_.score_best = _p_.score_new

        if _p_.score_new > _cand_.score_best:
            _p_.pos_current = _p_.pos_new
            _p_.score_current = _p_.score_new

            _cand_.pos_best = _p_.pos_new
            _cand_.score_best = _p_.score_new

            self._pbar_.best_since_iter = _cand_.i

    def _optimizer_eval(self, _cand_, _p_):
        _p_.score_new = _cand_.eval_pos(_p_.pos_new)
        self._pbar_.update_p_bar(1, _cand_)

    def _init_base_positioner(self, _cand_, positioner=None):
        if positioner:
            _p_ = positioner(**self._opt_args_.kwargs_opt)
        else:
            _p_ = BasePositioner(**self._opt_args_.kwargs_opt)

        _p_.pos_new = _cand_.pos_best
        _p_.score_new = _cand_.score_best

        return _p_
