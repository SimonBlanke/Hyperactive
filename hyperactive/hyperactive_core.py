# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .opt_args import Arguments
from .distribution import Distribution

from .search import Search


class HyperactiveCore:
    def __init__(self, _main_args_):
        self._main_args_ = _main_args_
        self._opt_args_ = Arguments(**self._main_args_.opt_para)
        self._opt_args_.set_opt_args(_main_args_.n_iter)

    def run(self):
        dist = Distribution()
        dist.dist(Search, self._main_args_, self._opt_args_)

        self.results = dist.results
        self.pos_list = dist.pos
        # self.para_list = None
        self.score_list = dist.scores

        self.eval_times = dist.eval_times
        self.iter_times = dist.iter_times
        self.best_scores = dist.best_scores
