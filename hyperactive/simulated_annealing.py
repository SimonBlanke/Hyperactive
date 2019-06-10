# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer
from .search_space import SearchSpace
from .model import MachineLearner
from .model import DeepLearner


class SimulatedAnnealing_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        scoring="accuracy",
        memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
        eps=1,
        t_rate=0.99,
    ):
        super().__init__(
            search_config,
            n_iter,
            scoring,
            memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            start_points,
        )

        self.eps = eps
        self.t_rate = t_rate

        self.temp = 0.1

        self.annealing_search_space = SearchSpace(start_points, search_config)

    def _get_neighbor_model(self, hyperpara_indices):
        hyperpara_indices_new = {}

        for hyperpara_name in hyperpara_indices:
            n_values = len(self.annealing_search_space.search_space[hyperpara_name])
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

        if self.model_type == "sklearn" or self.model_type == "xgboost":
            model_str = self._get_sklearn_model(n_process)
            self.annealing_search_space.create_mlSearchSpace(self.search_config)
            self.model = MachineLearner(
                self.search_config, self.scoring, self.cv, model_str
            )
        elif self.model_type == "keras":
            self.annealing_search_space.create_kerasSearchSpace(self.search_config)
            self.model = DeepLearner(self.search_config, self.scoring, self.cv)

        self._set_random_seed(n_process)
        n_steps = self._set_n_steps(n_process)

        score = 0
        score_best = 0
        score_current = 0

        hyperpara_indices = 0
        hyperpara_indices_best = 0
        hyperpara_indices_current = 0

        hyperpara_indices_current = self.annealing_search_space.init_eval(n_process)
        hyperpara_dict_current = self.annealing_search_space.pos_dict2values_dict(
            hyperpara_indices_current
        )
        score_current, train_time, sklearn_model = self.model.train_model(
            hyperpara_dict_current, X_train, y_train
        )

        score_best = score_current
        hyperpara_indices_best = hyperpara_indices_current

        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):
            self.temp = self.temp * self.t_rate
            rand = random.randint(0, 1)

            hyperpara_indices = self._get_neighbor_model(hyperpara_indices_current)
            hyperpara_dict = self.annealing_search_space.pos_dict2values_dict(
                hyperpara_indices
            )
            score, train_time, sklearn_model = self.model.train_model(
                hyperpara_dict, X_train, y_train
            )

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

        best_hyperpara_dict = self.annealing_search_space.pos_dict2values_dict(
            hyperpara_indices_best
        )
        score_best, train_time, sklearn_model = self.model.train_model(
            best_hyperpara_dict, X_train, y_train
        )

        if self.model_type == "sklearn" or self.model_type == "xgboost":
            start_point = self.model.create_start_point(best_hyperpara_dict, n_process)
        elif self.model_type == "keras":
            pass

        return sklearn_model, score_best, start_point
