# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .main_args import MainArgs

from .search import Search
from .search_process import SearchProcess

from .verbosity import Verbosity


def check_parameter(kwargs):
    pass


class Optimizer:
    def __init__(
        self, verbosity=3, warnings=False, ext_warnings=False,
    ):
        self.verb = Verbosity(verbosity, warnings)
        self.search_processes = []

    def add_search(
        self,
        **kwargs,
        # obj_func,
        # search_space,
        # obj_func_para=None,
        # memory="long",
        # optimizer="RandomSearch",
        # n_iter=10,
        # n_jobs=1,
        # random_state=1,
        # init=None,
        # distribution=None,
    ):
        # para_dict = check_parameter(args, kwargs)
        para_dict = kwargs
        self.n_jobs = para_dict["n_jobs"]

        for nth_process in range(self.n_jobs):
            new_search_process = SearchProcess(para_dict, self.verb)
            self.search_processes.append(new_search_process)

        self.search = Search(self.search_processes, para_dict, self.n_jobs)

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
