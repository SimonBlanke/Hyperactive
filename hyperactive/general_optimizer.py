# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .main_args import MainArgs

from .search import Search
from .search_process import SearchProcess

from .verbosity import Verbosity


def check_parameter(kwargs):
    pass


def set_default(args, kwargs):
    kwargs.setdefault("function_parameter", None)
    kwargs.setdefault("memory", "long")
    kwargs.setdefault("optimizer", "RandomSearch")
    kwargs.setdefault("n_iter", 10)
    kwargs.setdefault("n_jobs", 1)
    kwargs.setdefault("random_state", None)
    kwargs.setdefault("init", None)
    kwargs.setdefault("distribution", None)

    return kwargs


class Optimizer:
    def __init__(
        self, verbosity=3, warnings=False, ext_warnings=False,
    ):
        self.verb = Verbosity(verbosity, warnings)
        self.search_processes = []

    def add_search(self, *args, **kwargs):
        # check_parameter(args, kwargs)
        search_para = set_default(args, kwargs)
        for arg in args:
            if callable(arg):
                search_para["objective_function"] = arg
            elif isinstance(arg, dict):
                search_para["search_space"] = arg

        self.n_jobs = search_para["n_jobs"]

        for nth_process in range(self.n_jobs):
            new_search_process = SearchProcess(search_para, self.verb)
            self.search_processes.append(new_search_process)

        self.search = Search(self.search_processes, search_para, self.n_jobs)

    def run(self, max_time=None):
        self.search.run()

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
