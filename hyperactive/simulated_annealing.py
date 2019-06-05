# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer


class SimulatedAnnealing_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_space,
        n_iter,
        scoring="accuracy",
        tabu_memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
        eps=1,
        t_rate=0.95,
    ):
        super().__init__(
            search_space,
            n_iter,
            scoring,
            tabu_memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            start_points,
        )
        self._search = self._start_simulated_annealing

        self.eps = eps
        self.t_rate = t_rate

        self.temp = 0.1

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

    def _start_simulated_annealing(self, n_process):
        score = 0
        score_best = 0
        score_current = 0

        hyperpara_indices = 0
        hyperpara_indices_best = 0
        hyperpara_indices_current = 0

        self._set_random_seed(n_process)
        n_steps = int(self.n_iter / self.n_jobs)

        hyperpara_indices_current = self._init_eval(n_process)
        hyperpara_dict_current = self._pos_dict2values_dict(hyperpara_indices_current)
        score_current, train_time, sklearn_model = self._train_model(
            hyperpara_dict_current
        )

        score_best = score_current
        hyperpara_indices_best = hyperpara_indices_current

        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):
            self.temp = self.temp * self.t_rate
            rand = random.randint(0, 1)

            hyperpara_indices = self._get_neighbor_model(hyperpara_indices_current)
            hyperpara_dict = self._pos_dict2values_dict(hyperpara_indices)
            score, train_time, sklearn_model = self._train_model(hyperpara_dict)

            # Normalized score difference to have a factor for later use with temperature and random
            score_diff_norm = (score_current - score) / (score_current + score)

            if score > score_current:
                score_current = score
                hyperpara_indices_current = hyperpara_indices

                if score > score_best:
                    score_best = score
                    hyperpara_indices_best = hyperpara_indices

            elif np.exp(score_diff_norm / self.temp) > rand:
                score_current = score
                hyperpara_indices_current = hyperpara_indices

        hyperpara_dict_best = self._pos_dict2values_dict(hyperpara_indices_best)
        score_best, train_time, sklearn_model = self._train_model(hyperpara_dict_best)

        return sklearn_model, score_best, hyperpara_dict_best, train_time
