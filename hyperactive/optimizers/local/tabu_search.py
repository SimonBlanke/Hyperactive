# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class TabuOptimizer(BaseOptimizer):
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
        eps=1,
        tabu_memory=[3, 6, 9],
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
        self.tabu_memory = tabu_memory

        self.initializer = self._init_tabu

    def _memorize(self, _cand_, _p_):
        _p_.tabu_memory_short.append(_cand_.pos_best)
        _p_.tabu_memory_mid.append(_cand_.pos_best)
        _p_.tabu_memory_long.append(_cand_.pos_best)

        if len(_p_.tabu_memory_mid) > self.tabu_memory[0]:
            del _p_.tabu_memory_mid[0]

        if len(_p_.tabu_memory_short) > self.tabu_memory[1]:
            del _p_.tabu_memory_short[0]

        if len(_p_.tabu_memory_long) > self.tabu_memory[2]:
            del _p_.tabu_memory_long[0]

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

            self._memorize(_cand_, _p_)

        return _cand_

    def _init_tabu(self, _cand_, X, y):
        return super()._initialize(_cand_, X, y, TabuPositioner, pos_para=self.pos_para)


class TabuPositioner(BasePositioner):
    def __init__(self, eps=1):
        super().__init__(eps)

        self.tabu_memory_short = []
        self.tabu_memory_mid = []
        self.tabu_memory_long = []

    def climb_tabu(self, _cand_, eps_mod=1):
        in_tabu_mem = True
        while in_tabu_mem:
            pos_new = self.move_climb(_cand_)

            if not any((pos_new == pos).all() for pos in self.tabu_memory_short):
                in_tabu_mem = False
