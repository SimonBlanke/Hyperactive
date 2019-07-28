# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor


from ...base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xi = 0.01
        # Gaussian process with Mat??rn kernel as surrogate model
        # m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # self.gpr1 = GaussianProcessRegressor(kernel=kernel, alpha=0.02)
        self.gpr = GaussianProcessRegressor(
            kernel=self._arg_.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            # random_state=self._random_state,
        )

    def expected_improvement(self, xi=0.01):
        mu, sigma = self.gpr.predict(self.all_pos_comb, return_std=True)
        mu_sample = self.gpr.predict(self.X_sample)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide="warn"):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _all_possible_pos(self, cand):
        pos_space = []
        for dim_ in cand._space_.dim:
            pos_space.append(np.arange(dim_ + 1))

        self.n_dim = len(pos_space)
        self.all_pos_comb = np.array(np.meshgrid(*pos_space)).T.reshape(-1, self.n_dim)

    def propose_location(self, cand):
        self.gpr.fit(self.X_sample, self.Y_sample)
        ei = self.expected_improvement()
        ei = ei[:, 0]

        index_best = list(ei.argsort()[::-1])

        all_pos_comb_sorted = self.all_pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    def _iterate(self, i, _cand_, _p_, X, y):
        _p_.pos_new = self.propose_location(_cand_)
        _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

        self.X_sample = np.vstack((self.X_sample, _cand_.pos_best))
        self.Y_sample = np.vstack((self.Y_sample, _cand_.score_best))

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        _p_ = Bayesian()

        self._all_possible_pos(_cand_)
        self.X_sample = _cand_.pos_best.reshape(1, -1)
        self.Y_sample = _cand_.score_best.reshape(1, -1)

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_


class Bayesian:
    def __init__(self):
        pass
