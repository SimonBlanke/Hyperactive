# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .distribution import Distribution

from .search import Search


class HyperactiveCore:
    def __init__(self, _main_args_):
        self._main_args_ = _main_args_

    def run(self):
        dist = Distribution()
        dist.dist(Search, self._main_args_)

        self.results = dist.results
        self.pos_list = dist.pos
        # self.para_list = None
        self.score_list = dist.scores

        self.eval_times = dist.eval_times
        self.iter_times = dist.iter_times
        self.best_scores = dist.best_scores
