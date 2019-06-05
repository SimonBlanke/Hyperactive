# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm

from .base import BaseOptimizer


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

        self._search = self._start_random_search

    def _start_random_search(self, n_process):
        self._set_random_seed(n_process)
        n_steps = int(self.n_iter / self.n_jobs)

        best_model = None
        best_score = 0
        best_hyperpara_dict = None
        best_train_time = None

        hyperpara_indices = self._init_eval(n_process)
        hyperpara_dict = self._pos_dict2values_dict(hyperpara_indices)
        score, train_time, sklearn_model = self._train_model(hyperpara_dict)

        if score > best_score:
            best_model = sklearn_model
            best_score = score
            best_hyperpara_dict = hyperpara_dict
            best_train_time = train_time

        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):
            hyperpara_indices = self._get_random_position()

            hyperpara_dict = self._pos_dict2values_dict(hyperpara_indices)
            score, train_time, sklearn_model = self._train_model(hyperpara_dict)

            if score > best_score:
                best_model = sklearn_model
                best_score = score
                best_hyperpara_dict = hyperpara_dict
                best_train_time = train_time

        return best_model, best_score, best_hyperpara_dict, best_train_time
