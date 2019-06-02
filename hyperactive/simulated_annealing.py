# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
import tqdm

from .base import BaseOptimizer


class SimulatedAnnealing_Optimizer(BaseOptimizer):
    def __init__(self, search_dict, n_iter, scoring, eps=1, t_rate=0.9, n_jobs=1, cv=5):
        super().__init__(search_dict, n_iter, scoring, n_jobs, cv)
        self._search = self._start_simulated_annealing

        self.search_dict = search_dict
        self.scoring = scoring
        self.n_iter = n_iter

        self.eps = eps
        self.t_rate = t_rate

        self.temp = 100

        self.best_model = None

        self.score = 0
        self.score_best = 0
        self.score_current = 0

        self.hyperpara_indices = 0
        self.hyperpara_indices_best = 0
        self.hyperpara_indices_current = 0

    def _get_neighbor_model(self, hyperpara_indices):
        hyperpara_indices_new = {}

        for hyperpara_name in hyperpara_indices:
            n_values = len(self.hyperpara_search_dict[hyperpara_name])
            rand_eps = random.randint(-self.eps, self.eps + 1)

            index = hyperpara_indices[hyperpara_name]
            index_new = index + rand_eps
            hyperpara_indices_new[hyperpara_name] = index_new

            # don't go out of range
            if index_new < 0:
                index_new = 0
            if index_new > n_values - 1:
                index_new = n_values - 1

            hyperpara_indices_new[hyperpara_name] = index_new

        return hyperpara_indices_new

    def _start_simulated_annealing(self, n_iter):
        n_steps = int(self.n_iter / self.n_jobs)

        self.hyperpara_indices_current = self._get_random_position()
        self.hyperpara_dict_current = self._pos_dict2values_dict(
            self.hyperpara_indices_current
        )
        self.score_current, train_time, sklearn_model = self._train_model(
            self.hyperpara_dict_current
        )

        self.score_best = self.score_current
        self.hyperpara_indices_best = self.hyperpara_indices_current

        for i in tqdm.tqdm(range(n_steps), position=n_iter, leave=False):
            self.temp = self.temp * self.t_rate
            rand = random.randint(0, 1)

            self.hyperpara_indices = self._get_neighbor_model(
                self.hyperpara_indices_current
            )
            self.hyperpara_dict = self._pos_dict2values_dict(self.hyperpara_indices)
            self.score, train_time, sklearn_model = self._train_model(
                self.hyperpara_dict
            )

            # Normalized score difference to have a factor for later use with temperature and random
            score_diff_norm = (self.score_current - self.score) / (
                self.score_current + self.score
            )

            if self.score > self.score_current:
                self.score_current = self.score
                self.hyperpara_indices_current = self.hyperpara_indices

                if self.score > self.score_best:
                    self.score_best = self.score
                    self.hyperpara_indices_best = self.hyperpara_indices

            elif np.exp(score_diff_norm / self.temp) > rand:
                self.score_current = self.score
                self.hyperpara_indices_current = self.hyperpara_indices

        hyperpara_dict_best = self._pos_dict2values_dict(self.hyperpara_indices_best)
        score_best, train_time, sklearn_model = self._train_model(hyperpara_dict_best)

        return sklearn_model, score_best, hyperpara_dict_best, train_time
