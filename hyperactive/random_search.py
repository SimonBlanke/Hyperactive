# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm

from .base import BaseOptimizer
from .base import BaseCandidate
from .search_space import SearchSpace


class RandomSearch_Optimizer(BaseOptimizer):
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

        self.space = SearchSpace(warm_start, search_config)

    def _search(self, n_process, X_train, y_train):
        hyperpara_indices = self._init_search(n_process, X_train, y_train)
        self._set_random_seed(n_process)
        self.n_steps = self._set_n_steps(n_process)
        self.candidate = BaseCandidate(self.model)

        hyperpara_dict = self.space.pos_dict2values_dict(hyperpara_indices)

        self.candidate.set_position(hyperpara_dict)
        self.candidate.eval(X_train, y_train)

        self.best_score = self.candidate.score
        self.best_hyperpara_dict = self.candidate.hyperpara_dict
        self.best_model = self.candidate.sklearn_model

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

            hyperpara_indices = self.space.get_random_position()
            hyperpara_dict = self.space.pos_dict2values_dict(hyperpara_indices)

            self.candidate.set_position(hyperpara_dict)
            self.candidate.eval(X_train, y_train)

            if self.candidate.score > self.best_score:
                self.best_score = self.candidate.score
                self.best_hyperpara_dict = self.candidate.hyperpara_dict
                self.best_model = self.candidate.sklearn_model

        start_point = self._finish_search(self.best_hyperpara_dict, n_process)

        return self.best_model, self.best_score, start_point

    def _search_best_loss(self, n_process, X_train, y_train):
        for i in tqdm.tqdm(
            range(self.n_steps),
            desc=str(self.model_str),
            position=n_process,
            leave=False,
        ):

            hyperpara_indices = self.space.get_random_position()
            hyperpara_dict = self.space.pos_dict2values_dict(hyperpara_indices)

            self.candidate.set_position(hyperpara_dict)
            self.candidate.eval(X_train, y_train)

            if self.candidate.score < self.best_score:
                self.best_score = self.candidate.score
                self.best_hyperpara_dict = self.candidate.hyperpara_dict
                self.best_model = self.candidate.sklearn_model

        start_point = self._finish_search(self.best_hyperpara_dict, n_process)

        return self.best_model, self.best_score, start_point
