# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .base import BaseOptimizer


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
        hyperband_init=False,
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
            hyperband_init,
        )

        self.xi = 0.01

        # Gaussian process with Mat??rn kernel as surrogate model
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52, alpha=0.1)

    def expected_improvement(self, all_pos_comb, X_sample, Y_sample, xi=0.01):
        # print("\n ei: all_pos_comb     ", all_pos_comb)
        print("\n ei: X_sample", X_sample)
        # print("\n ei: Y_sample", Y_sample)

        mu, sigma = self.gpr.predict(all_pos_comb, return_std=True)
        mu_sample = self.gpr.predict(X_sample)

        # print("\nsigma", sigma.shape)
        # print("\nmu", mu.shape)
        # print("\nmu_sample", mu_sample.shape)

        sigma = sigma.reshape(-1, 1)
        # print("\nsigma", sigma.shape)

        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide="warn"):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            # print("\nimp", imp.shape)
            # print("\nZ", Z.shape)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _all_possible_pos(self, cand):
        n_dim = cand._space_.dim.size

        pos_space = []
        for dim_ in cand._space_.dim:
            pos_space.append(np.arange(dim_))

        # print("pos_space", pos_space)

        all_pos_comb = np.array(np.meshgrid(*pos_space)).T.reshape(-1, n_dim)

        return all_pos_comb

    def propose_location(self, cand, X_sample, Y_sample):
        all_pos_comb = self._all_possible_pos(cand).reshape(-1, 2)

        # print("all_pos_comb", all_pos_comb, all_pos_comb.shape)

        ei = self.expected_improvement(all_pos_comb, X_sample, Y_sample)

        ei = ei[:, 0]

        # print("ei", ei[:10], ei.shape)
        # print("all_pos_comb", all_pos_comb[:10], all_pos_comb.shape)

        index_best = list(ei.argsort()[::-1])

        # print("index_best", index_best)

        all_pos_comb_sorted = all_pos_comb[index_best]

        # print("all_pos_comb_sorted", all_pos_comb_sorted[:10])

        # all_pos_comb_sorted, _ = self._sort_for_best(all_pos_comb, ei)

        # print("\nall_pos_comb_sorted", all_pos_comb_sorted.shape)

        pos_best = all_pos_comb_sorted[0]

        # print("all_pos_comb_sorted", all_pos_comb_sorted)

        # print("pos_best", pos_best)

        return pos_best

    def _move(self, cand, X_sample, Y_sample):
        pos = self.propose_location(cand, X_sample, Y_sample)
        cand.pos = pos

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        X_sample = _cand_.pos.reshape(1, -1)
        Y_sample = _cand_.score.reshape(1, -1)

        """
        _cand_.pos = _cand_._space_.get_random_position()
        _cand_.eval(X, y)

        X_sample = np.vstack((X_sample, _cand_.pos))
        Y_sample = np.vstack((Y_sample, _cand_.score))
        """

        for i in tqdm.tqdm(**self._tqdm_dict(_cand_)):
            self.gpr.fit(X_sample, Y_sample)

            self._move(_cand_, X_sample, Y_sample)
            _cand_.eval(X, y)

            # print("_cand_.pos", _cand_.pos)

            if _cand_.score > _cand_.score_best:
                _cand_.score_best = _cand_.score
                _cand_.pos_best = _cand_.pos

            X_sample = np.vstack((X_sample, _cand_.pos))
            Y_sample = np.vstack((Y_sample, _cand_.score))

        return _cand_
