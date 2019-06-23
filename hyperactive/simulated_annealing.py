# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer
from .base import BaseCandidate
from .search_space import SearchSpace


class SimulatedAnnealing_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        eps=1,
        t_rate=0.99,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
        )

        self.eps = eps
        self.t_rate = t_rate

        self.temp = 0.1

        self.space = SearchSpace(warm_start, search_config)

    def _get_neighbor_model(self, hyperpara_indices):
        hyperpara_indices_new = {}

        for hyperpara_name in hyperpara_indices:
            n_values = len(self.space.search_space[hyperpara_name])
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

    def _search(self, n_process, X_train, y_train):
        self.hyperpara_indices_current = self._init_search(n_process, X_train, y_train)
        self._set_random_seed(n_process)
        self.n_steps = self._set_n_steps(n_process)
        self.candidate = BaseCandidate(self.model)

        hyperpara_dict = self.space.pos_dict2values_dict(
            self.hyperpara_indices_current
        )

        self.candidate.set_position(hyperpara_dict)
        self.candidate.eval(X_train, y_train)

        self.best_score = self.candidate.score
        self.best_hyperpara_dict = self.candidate.hyperpara_dict
        self.best_hyperpara_indices = self.hyperpara_indices_current
        self.best_model = self.candidate.sklearn_model

        self.score_current = self.candidate.score

        if self.metric_type == "score":
            return self._search_best_score(n_process, X_train, y_train)
        elif self.metric_type == "loss":
            return self._search_best_loss(n_process, X_train, y_train)

    def _search_best_score(self, n_process, X_train, y_train):
        for i in tqdm.tqdm(
            range(self.n_steps),
            desc=str(self.model_str),
            position=n_process,
            leave=False,
        ):
            self.temp = self.temp * self.t_rate
            rand = random.randint(0, 1)

            hyperpara_indices = self._get_neighbor_model(self.hyperpara_indices_current)
            hyperpara_dict = self.space.pos_dict2values_dict(
                hyperpara_indices
            )

            self.candidate.set_position(hyperpara_dict)
            self.candidate.eval(X_train, y_train)

            # Normalized score difference to have a factor for later use with temperature and random
            score_diff_norm = (self.score_current - self.candidate.score) / (
                self.score_current + self.candidate.score
            )

            if self.candidate.score > self.score_current:
                self.score_current = self.candidate.score
                self.hyperpara_indices_current = hyperpara_indices

                if self.candidate.score > self.best_score:
                    self.best_score = self.candidate.score
                    self.best_hyperpara_indices = hyperpara_indices

            elif np.exp(score_diff_norm / self.temp) > rand:
                self.score_current = self.candidate.score
                self.hyperpara_indices_current = hyperpara_indices

        self.best_hyperpara_dict = self.space.pos_dict2values_dict(
            self.best_hyperpara_indices
        )
        best_score, train_time, sklearn_model = self.model.train_model(
            self.best_hyperpara_dict, X_train, y_train
        )

        start_point = self._finish_search(self.best_hyperpara_dict, n_process)

        return sklearn_model, best_score, start_point

    def _search_best_loss(self, n_process, X_train, y_train):
        for i in tqdm.tqdm(
            range(self.n_steps),
            desc=str(self.model_str),
            position=n_process,
            leave=False,
        ):
            self.temp = self.temp * self.t_rate
            rand = random.randint(0, 1)

            hyperpara_indices = self._get_neighbor_model(self.hyperpara_indices_current)
            hyperpara_dict = self.space.pos_dict2values_dict(
                hyperpara_indices
            )

            self.candidate.set_position(hyperpara_dict)
            self.candidate.eval(X_train, y_train)

            # Normalized score difference to have a factor for later use with temperature and random
            score_diff_norm = (self.score_current - self.candidate.score) / (
                self.score_current + self.candidate.score
            )

            if self.candidate.score < self.score_current:
                self.score_current = self.candidate.score
                self.hyperpara_indices_current = hyperpara_indices

                if self.candidate.score < self.best_score:
                    self.best_score = self.candidate.score
                    self.best_hyperpara_indices = hyperpara_indices

            elif np.exp(score_diff_norm / self.temp) > rand:
                self.score_current = self.candidate.score
                self.hyperpara_indices_current = hyperpara_indices

        self.best_hyperpara_dict = self.space.pos_dict2values_dict(
            self.best_hyperpara_indices
        )
        best_score, train_time, sklearn_model = self.model.train_model(
            self.best_hyperpara_dict, X_train, y_train
        )

        start_point = self._finish_search(self.best_hyperpara_dict, n_process)

        return sklearn_model, best_score, start_point
