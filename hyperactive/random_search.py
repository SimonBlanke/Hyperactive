"""
MIT License

Copyright (c) [2018] [Simon Blanke]
Email: simon.blanke@yahoo.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
