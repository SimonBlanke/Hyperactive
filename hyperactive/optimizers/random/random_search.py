# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)

    def _iterate(self, i, _cand_):
        self._p_.move_random(_cand_)
        self._optimizer_eval(_cand_, self._p_)

        self._update_pos(_cand_, self._p_)

        return _cand_

    def _init_iteration(self, _cand_):
        self._p_ = super()._init_base_positioner(_cand_)

        self._optimizer_eval(_cand_, self._p_)
        self._update_pos(_cand_, self._p_)

    def _finish_search(self):
        self._pbar_.close_p_bar()

        return self._p_
