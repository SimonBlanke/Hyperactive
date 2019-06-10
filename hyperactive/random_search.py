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
        memory=None,
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
            memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            start_points,
        )

        self.search_config = search_config
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

        self.random_search_space = SearchSpace(start_points, search_config)

    def _search(self, n_process, X_train, y_train):
        model_str = self._get_sklearn_model(n_process)

        if self.model_type == "sklearn" or self.model_type == "xgboost":
            self.random_search_space.create_mlSearchSpace(self.search_config)
            self.model = MachineLearner(
                self.search_config, self.scoring, self.cv, model_str
            )
        elif self.model_type == "keras":
            self.random_search_space.create_kerasSearchSpace(self.search_config)
            self.model = DeepLearner(self.search_config, self.scoring, self.cv)

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

        if self.model_type == "sklearn" or self.model_type == "xgboost":
            start_point = self.model.create_start_point(best_hyperpara_dict, n_process)
        elif self.model_type == "keras":
            pass

        return best_model, best_score, start_point
