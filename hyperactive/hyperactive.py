# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import multiprocessing


from .checks import check_args
from .search import Search
from .search_process import SearchProcess
from .search_process_info import SearchProcessInfo


def set_n_jobs(n_jobs):
    """Sets the number of jobs to run in parallel"""
    num_cores = multiprocessing.cpu_count()
    if n_jobs == -1 or n_jobs > num_cores:
        return num_cores
    else:
        return n_jobs


def get_class(file_path, class_name):
    module = import_module(file_path, "hyperactive")
    return getattr(module, class_name)


def no_ext_warnings():
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn


class Hyperactive:
    def __init__(
        self,
        X,
        y,
        random_state=None,
        verbosity={
            "print_search_info": True,
            "progress_bar": True,
            "print_results": True,
        },
        ext_warnings=False,
    ):
        self.info = SearchProcessInfo(X, y, random_state, verbosity)

    def add_search(
        self,
        model,
        search_space,
        n_iter,
        name=None,
        optimizer="RandomSearch",
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        memory="short",
    ):
        """
        check_args(
            model, search_space, n_iter, name, optimizer, n_jobs, initialize, memory,
        )
        """
        n_jobs = set_n_jobs(n_jobs)

        for nth_job in range(n_jobs):
            nth_process = len(self.info.process_infos)

            self.info.add_search_process(
                nth_process,
                model,
                search_space,
                n_iter,
                name,
                optimizer,
                initialize,
                memory,
            )

    def run(self, max_time=None, distribution=None):
        self.search = Search(1, 1)
        self.info.add_run_info(max_time, distribution)

        start_time = time.time()
        self.search.run(self.info.process_infos)

        """
        self.eval_times = self.search.eval_times_dict
        self.iter_times = self.search.iter_times_dict

        self.positions = self.search.positions_dict
        self.scores = self.search.scores_dict
        self.score_best_list = self.search.best_score_list_dict

        self.para_best = self.search.para_best_dict
        self.score_best = self.search.score_best_dict

        self.position_results = self.search.position_results
        """

