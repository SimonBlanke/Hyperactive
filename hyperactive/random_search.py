# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm

from .base import BaseOptimizer
from .search_space import SearchSpace
from .model import MachineLearner
from .model import DeepLearner


class RandomSearch_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        scoring="accuracy",
        tabu_memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
    ):
        super().__init__(
            search_config,
            n_iter,
            scoring,
            tabu_memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            start_points,
        )

        self.random_search_space = SearchSpace(start_points, search_config)

        if self.model_type == "sklearn":
            self.random_search_space.create_mlSearchSpace(search_config)
            self.model = MachineLearner(search_config, scoring, cv)
        elif self.model_type == "keras":
            self.random_search_space.create_kerasSearchSpace(search_config)
            self.model = DeepLearner(search_config, scoring, cv)

    def _search(self, n_process, X_train, y_train):
        self._set_random_seed(n_process)
        n_steps = self._set_n_steps(n_process)

        best_model = None
        best_score = 0
        best_hyperpara_dict = None
        best_train_time = None

        hyperpara_indices = self.random_search_space.init_eval(n_process)

        hyperpara_dict = self.random_search_space.pos_dict2values_dict(
            hyperpara_indices
        )

        score, train_time, sklearn_model = self.model.train_model(
            hyperpara_dict, X_train, y_train
        )

        if score > best_score:
            best_model = sklearn_model
            best_score = score
            best_hyperpara_dict = hyperpara_dict
            best_train_time = train_time

        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):

            hyperpara_indices = self.random_search_space.get_random_position()
            hyperpara_dict = self.random_search_space.pos_dict2values_dict(
                hyperpara_indices
            )
            score, train_time, sklearn_model = self.model.train_model(
                hyperpara_dict, X_train, y_train
            )

            if score > best_score:
                best_model = sklearn_model
                best_score = score
                best_hyperpara_dict = hyperpara_dict
                best_train_time = train_time

        return best_model, best_score, best_hyperpara_dict, best_train_time
