# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm

from .base import BaseOptimizer
from .base import SearchSpace
from .base import MachineLearner


class RandomSearch_Optimizer(BaseOptimizer):
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

        self.search_space = SearchSpace(start_points, search_space)
        self.machine_learner = MachineLearner(search_space, scoring, cv)

    def _search(self, n_process, X_train, y_train):
        self._set_random_seed(n_process)
        n_steps = self._set_n_steps(n_process)

        best_model = None
        best_score = 0
        best_hyperpara_dict = None
        best_train_time = None

        hyperpara_indices = self.search_space.init_eval(n_process)

        hyperpara_dict = self.search_space.pos_dict2values_dict(hyperpara_indices)
        score, train_time, sklearn_model = self.machine_learner.train_model(
            hyperpara_dict, X_train, y_train
        )

        if score > best_score:
            best_model = sklearn_model
            best_score = score
            best_hyperpara_dict = hyperpara_dict
            best_train_time = train_time

        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):

            hyperpara_indices = self.search_space.get_random_position()
            hyperpara_dict = self.search_space.pos_dict2values_dict(hyperpara_indices)
            score, train_time, sklearn_model = self.machine_learner.train_model(
                hyperpara_dict, X_train, y_train
            )

            if score > best_score:
                best_model = sklearn_model
                best_score = score
                best_hyperpara_dict = hyperpara_dict
                best_train_time = train_time

        return best_model, best_score, best_hyperpara_dict, best_train_time
