# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class TabuOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_para = {"eps": self._arg_.eps}

    def _memorize(self, _cand_, _p_):
        for i in range(3):
            _p_.tabu_memory_[i].append(_cand_.pos_best)

            if len(_p_.tabu_memory_[i]) > self._arg_.tabu_memory[0]:
                del _p_.tabu_memory_[i][0]

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = _p_.climb_tabu(_cand_, _p_.pos_current)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

            self._memorize(_cand_, _p_)

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        return super()._init_base_positioner(
            _cand_, TabuPositioner, pos_para=self.pos_para
        )


class TabuPositioner(BasePositioner):
    def __init__(self, eps=1):
        super().__init__(eps)

        tabu_memory_short = []
        tabu_memory_mid = []
        tabu_memory_long = []

        self.tabu_memory_ = [tabu_memory_short, tabu_memory_mid, tabu_memory_long]

    def climb_tabu(self, _cand_, pos, eps_mod=1):
        in_tabu_mem = True
        pos_new = None

        while in_tabu_mem:
            pos_new = self.move_climb(_cand_, pos)

            for i in range(3):
                if not any((pos_new == pos).all() for pos in self.tabu_memory_[i]):
                    in_tabu_mem = False

        return pos_new
