# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomRestartHillClimbingOptimizer(HillClimbingOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)
        self.n_iter_restart = int(self._main_args_.n_iter / self._opt_args_.n_restarts)

    def _iterate(self, i, _cand_, _p_):
        _cand_, _p_ = self._hill_climb_iter(i, _cand_, _p_)

        if self.n_iter_restart != 0 and i % self.n_iter_restart == 0:
            _p_.pos_current = _p_.move_random(_cand_)

        return _cand_
