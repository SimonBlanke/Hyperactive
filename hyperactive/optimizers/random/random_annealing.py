# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..local import HillClimbingOptimizer


class RandomAnnealingOptimizer(HillClimbingOptimizer):
    def __init__(self, _opt_args_):
        super().__init__(_opt_args_)
        self.temp = _opt_args_.start_temp

    def _iterate(self, i, _cand_):
        self.p_list[0].move_climb(
            _cand_, self.p_list[0].pos_current, epsilon_mod=self.temp / 10
        )
        self._optimizer_eval(_cand_, self.p_list[0])
        self._update_pos(_cand_, self.p_list[0])

        self.temp = self.temp * self._opt_args_.annealing_rate

        return _cand_
