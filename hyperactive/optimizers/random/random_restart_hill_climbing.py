# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base_optimizer import BaseOptimizer


class RandomRestartHillClimbingOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kwargs["eps"] = 1
        kwargs["n_restarts"] = 10

        self.pos_para = {"eps": kwargs["eps"]}
        self.n_restarts = kwargs["n_restarts"]

        self.n_iter_restart = int(self.n_iter / self.n_restarts)

        self.initializer = self._init_rr_climber

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

        if self.n_iter_restart != 0 and i % self.n_iter_restart == 0:
            _p_.pos_current = _p_.move_random(_cand_)

        return _cand_

    def _init_rr_climber(self, _cand_, X, y):
        return super()._initialize(_cand_, pos_para=self.pos_para)
