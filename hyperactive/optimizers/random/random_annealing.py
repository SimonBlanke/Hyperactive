# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base import BaseOptimizer
from ...base import BasePositioner


class RandomAnnealingOptimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        memory=True,
        scatter_init=False,
        eps=100,
        t_rate=0.98,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
            memory,
            scatter_init,
        )

        self.pos_para = {"eps": eps}
        self.t_rate = t_rate
        self.temp = 1

        self.initializer = self._init_rnd_annealing

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current, eps_mod=self.temp)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

        self.temp = self.temp * self.t_rate

        return _cand_

    def _init_rnd_annealing(self, _cand_, X, y):
        return super()._initialize(_cand_, X, y, pos_para=self.pos_para)
