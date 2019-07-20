# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# from sklearn.gaussian_process.kernels import ConstantKernel, Matern
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from ...base import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
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

        self.xi = 0.01
        self.initializer = self.init_bayesian

        # Gaussian process with Mat??rn kernel as surrogate model
        # m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # self.gpr1 = GaussianProcessRegressor(kernel=kernel, alpha=0.02)
        self.gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
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
        ei = self.expected_improvement()
        ei = ei[:, 0]

        index_best = list(ei.argsort()[::-1])

        all_pos_comb_sorted = self.all_pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    def _iterate(self, i, _cand_, X, y):
        self.gpr.fit(self.X_sample, self.Y_sample)
        pos = self.propose_location(_cand_)
        _cand_.pos = pos

        _cand_.eval(X, y)

        if _cand_.score > _cand_.score_best:
            _cand_.score_best = _cand_.score
            _cand_.pos_best = _cand_.pos

        self.X_sample = np.vstack((self.X_sample, _cand_.pos))
        self.Y_sample = np.vstack((self.Y_sample, _cand_.score))

        if self._show_progress_bar():
            self.p_bar.update(1)

        return _cand_

    def init_bayesian(self, _cand_):
        self._all_possible_pos(_cand_)
        self.X_sample = _cand_.pos.reshape(1, -1)
        self.Y_sample = _cand_.score.reshape(1, -1)
