# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .base import BaseOptimizer
import tqdm


class RandomSearch_Optimizer(BaseOptimizer):
    def __init__(self, ml_search_dict, n_searches, scoring, n_jobs=1, cv=5):
        super().__init__(ml_search_dict, n_searches, scoring, n_jobs, cv)
        self._search = self._start_random_search

    def _start_random_search(self, n_searches):
        n_steps = int(self.n_searches / self.n_jobs)

        best_model = None
        best_score = 0
        best_hyperpara_dict = None
        best_train_time = None

        for i in tqdm.tqdm(range(n_steps), position=n_searches, leave=False):
            hyperpara_indices = self._get_random_position()
            hyperpara_dict = self._pos_dict2values_dict(hyperpara_indices)
            score, train_time, sklearn_model = self._train_model(hyperpara_dict)

            if score > best_score:
                best_model = sklearn_model
                best_score = score
                best_hyperpara_dict = hyperpara_dict
                best_train_time = train_time

        return best_model, best_score, best_hyperpara_dict, best_train_time
