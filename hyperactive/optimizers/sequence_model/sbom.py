# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np


from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class SBOM(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)

    def _all_possible_pos(self, cand):
        pos_space = []
        for dim_ in cand._space_.dim:
            pos_space.append(np.arange(dim_ + 1))

        self.n_dim = len(pos_space)
        self.all_pos_comb = np.array(np.meshgrid(*pos_space)).T.reshape(-1, self.n_dim)

    def _init_iteration(self, _cand_):
        self._p_ = SbomPositioner()
        self._p_.pos_new = self._p_.move_random(_cand_)

        self._optimizer_eval(_cand_, self._p_)
        self._update_pos(_cand_, self._p_)

        self._all_possible_pos(_cand_)

        if self._opt_args_.warm_start_smbo:
            self.X_sample = _cand_.mem._get_para()
            self.Y_sample = _cand_.mem._get_score()
        else:
            self.X_sample = _cand_.pos_best.reshape(1, -1)
            self.Y_sample = np.array(_cand_.score_best).reshape(1, -1)

    def _finish_search(self):
        self._pbar_.close_p_bar()

        return self._p_


class SbomPositioner(BasePositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
