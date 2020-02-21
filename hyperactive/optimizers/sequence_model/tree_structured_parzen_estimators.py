# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from sklearn.neighbors import KernelDensity

from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class TreeStructuredParzenEstimators(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)
        self.kd_best = KernelDensity()
        self.kd_worst = KernelDensity()

    def _get_samples(self):
        n_samples = self.X_sample.shape[0]

        n_best = int(n_samples * self._opt_args_.gamme_tpe)

        Y_sample = self.Y_sample[:, 0]
        index_best = Y_sample.argsort()[-n_best:][::-1]

        best_samples = self.X_sample[index_best]
        worst_samples = self.X_sample[~index_best]

        return best_samples, worst_samples

    def _all_possible_pos(self, cand):
        pos_space = []
        for dim_ in cand._space_.dim:
            pos_space.append(np.arange(dim_ + 1))

        self.n_dim = len(pos_space)
        self.all_pos_comb = np.array(np.meshgrid(*pos_space)).T.reshape(-1, self.n_dim)

    def expected_improvement(self):
        logprob_best = self.kd_best.score_samples(self.all_pos_comb)
        logprob_worst = self.kd_worst.score_samples(self.all_pos_comb)

        prob_best = np.exp(logprob_best)
        prob_worst = np.exp(logprob_worst)

        return np.divide(
            prob_best, prob_worst, out=np.zeros_like(prob_worst), where=prob_worst != 0
        )

    def propose_location(self, cand):
        best_samples, worst_samples = self._get_samples()

        self.kd_best.fit(best_samples)
        self.kd_worst.fit(worst_samples)

        exp_imp = self.expected_improvement()
        index_best = list(exp_imp.argsort()[::-1])

        all_pos_comb_sorted = self.all_pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    def _iterate(self, i, _cand_, _p_):
        if i < self._opt_args_.start_up_evals:
            _p_.pos_new = _p_.move_random(_cand_)
            _p_.score_new = _cand_.eval_pos(_p_.pos_new)

        else:
            _p_.pos_new = self.propose_location(_cand_)
            _p_.score_new = _cand_.eval_pos(_p_.pos_new)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

        self.X_sample = np.vstack((self.X_sample, _p_.pos_new))
        self.Y_sample = np.vstack((self.Y_sample, _p_.score_new))

        return _cand_

    def _init_opt_positioner(self, _cand_):
        _p_ = Bayesian()

        self._all_possible_pos(_cand_)

        if self._opt_args_.warm_start_smbo:
            self.X_sample = _cand_.mem._get_para()
            self.Y_sample = _cand_.mem._get_score()
        else:
            self.X_sample = _cand_.pos_best.reshape(1, -1)
            self.Y_sample = np.array(_cand_.score_best).reshape(1, -1)

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_


class Bayesian(BasePositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
