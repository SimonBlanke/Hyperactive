# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)

    def _iterate(self, i, _cand_, _p_):
        _p_.pos_new = _p_.move_random(_cand_)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

        return _cand_

    def _init_opt_positioner(self, _cand_):
        return super()._init_base_positioner(_cand_)
