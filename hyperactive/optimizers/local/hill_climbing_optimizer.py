# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class HillClimbingOptimizer(BaseOptimizer):
    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)

    def _hill_climb_iter(self, i, _cand_):
        score_new = -np.inf
        pos_new = None

        self._p_.move_climb(_cand_, self._p_.pos_current)
        self._optimizer_eval(_cand_, self._p_)

        if self._p_.score_new > score_new:
            score_new = self._p_.score_new
            pos_new = self._p_.pos_new

        if i % self._opt_args_.n_neighbours == 0:
            self._p_.pos_new = pos_new
            self._p_.score_new = score_new

            self._update_pos(_cand_, self._p_)

    def _iterate(self, i, _cand_):
        self._hill_climb_iter(i, _cand_)

        return _cand_

    def _init_iteration(self, _cand_):
        self._p_ = super()._init_base_positioner(
            _cand_, positioner=HillClimbingPositioner
        )

        self._optimizer_eval(_cand_, self._p_)
        self._update_pos(_cand_, self._p_)

    def _finish_search(self):
        self._pbar_.close_p_bar()

        return self._p_


class HillClimbingPositioner(BasePositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.epsilon = kwargs["epsilon"]
        self.climb_dist = kwargs["climb_dist"]
