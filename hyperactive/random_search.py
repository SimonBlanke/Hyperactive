# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm

from .base import BaseOptimizer
from .search_space import SearchSpace


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

        self.search_space_inst = SearchSpace(start_points, search_config)

    def _search(self, n_process, X_train, y_train):
        hyperpara_indices = self._init_search(n_process, X_train, y_train)
        self._set_random_seed(n_process)
        n_steps = self._set_n_steps(n_process)

        best_model = None
        best_score = 0
        best_hyperpara_dict = None
        best_train_time = None

        hyperpara_dict = self.search_space_inst.pos_dict2values_dict(hyperpara_indices)

        score, train_time, sklearn_model = self.model.train_model(
            hyperpara_dict, X_train, y_train
        )

        if score > best_score:
            best_model = sklearn_model
            best_score = score
            best_hyperpara_dict = hyperpara_dict
            best_train_time = train_time

        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):

            hyperpara_indices = self.search_space_inst.get_random_position()
            hyperpara_dict = self.search_space_inst.pos_dict2values_dict(
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

        start_point = self._finish_search(best_hyperpara_dict, n_process)

        return best_model, best_score, start_point
