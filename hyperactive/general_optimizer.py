# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from .search import Search
from .search_process import SearchProcess
from .process_arguments import ProcessArguments
from .verbosity import Verbosity


class Optimizer:
    def __init__(
        self, random_state=None, verbosity=3, warnings=False, ext_warnings=False,
    ):
        self.verb = Verbosity(verbosity, warnings)
        self.random_state = random_state
        self.search_processes = []

    def add_search(self, *args, **kwargs):
        pro_arg = ProcessArguments(args, kwargs, random_state=self.random_state)

        for nth_job in range(pro_arg.n_jobs):
            new_search_process = SearchProcess(nth_job, pro_arg, self.verb)
            self.search_processes.append(new_search_process)

        self.search = Search(self.search_processes)

    def run(self, max_time=None):
        if max_time is not None:
            max_time = max_time * 60

        start_time = time.time()

        self.search.run(start_time, max_time)

        """
        dist = Distribution()
        dist.dist(Search, self._main_args_)

        self.results = dist.results
        self.pos_list = dist.pos
        # self.para_list = None
        self.score_list = dist.scores

        self.eval_times = dist.eval_times
        self.iter_times = dist.iter_times
        self.best_scores = dist.best_scores
        """
