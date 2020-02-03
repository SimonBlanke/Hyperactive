# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from scipy.stats import norm


from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)
        self.gpr = self._opt_args_.gpr

    def expected_improvement(self):
        mu, sigma = self.gpr.predict(self.all_pos_comb)
        mu_sample, _ = self.gpr.predict(self.X_sample)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)

        imp = mu - mu_sample_opt - self._opt_args_.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma != 0)
        exp_imp = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        exp_imp[sigma == 0.0] = 0.0

        return exp_imp

    def _all_possible_pos(self, cand):
        pos_space = []
        for dim_ in cand._space_.dim:
            pos_space.append(np.arange(dim_ + 1))

        self.n_dim = len(pos_space)
        self.all_pos_comb = np.array(np.meshgrid(*pos_space)).T.reshape(-1, self.n_dim)

    def propose_location(self, cand):
        self.gpr.fit(self.X_sample, self.Y_sample)
        exp_imp = self.expected_improvement()
        exp_imp = exp_imp[:, 0]

        index_best = list(exp_imp.argsort()[::-1])

        all_pos_comb_sorted = self.all_pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    def _iterate(self, i, _cand_, _p_):
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
