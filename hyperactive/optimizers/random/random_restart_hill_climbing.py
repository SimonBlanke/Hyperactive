# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter_restart = int(self._core_.n_iter / self._arg_.n_restarts)

    def _iterate(self, i, _cand_, _p_, X, y):
        _cand_, _p_ = self._hill_climb_iter(_cand_, _p_, X, y)

        if self.n_iter_restart != 0 and i % self.n_iter_restart == 0:
            _p_.pos_current = _p_.move_random(_cand_)

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        return super()._init_base_positioner(_cand_)
