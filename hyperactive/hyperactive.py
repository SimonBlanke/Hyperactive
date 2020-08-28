# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import multiprocessing

from importlib import import_module

from .checks import check_args
from .search import Search


search_process_dict = {
    False: "SearchProcessNoMem",
    "short": "SearchProcessShortMem",
    "long": "SearchProcessLongMem",
}


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
        self, X, y, random_state=None, verbosity=1, warnings=False, ext_warnings=False,
    ):
        if ext_warnings is False:
            no_ext_warnings()

        self.training_data = {
            "features": X,
            "target": y,
        }
        self.verbosity = verbosity
        self.random_state = random_state
        self.search_processes = []

    def _add_process(
        self,
        nth_process,
        model,
        search_space,
        name,
        n_iter,
        optimizer,
        n_jobs,
        init_para,
        memory,
    ):
        search_process_kwargs = {
            "nth_process": nth_process,
            "model": model,
            "search_space": search_space,
            "search_name": name,
            "n_iter": n_iter,
            "training_data": self.training_data,
            "optimizer": optimizer,
            "n_jobs": n_jobs,
            "init_para": init_para,
            "memory": memory,
            "random_state": self.random_state,
            "verbosity": 1,
        }
        SearchProcess = get_class(".search_process", search_process_dict[memory])
        new_search_process = SearchProcess(**search_process_kwargs)
        self.search_processes.append(new_search_process)

    def add_search(
        self,
        model,
        search_space,
        name=None,
        n_iter=10,
        optimizer="RandomSearch",
        n_jobs=1,
        init_para=[],
        memory="short",
    ):
        check_args(
            model, search_space, n_iter, optimizer, n_jobs, init_para, memory,
        )

        n_jobs = set_n_jobs(n_jobs)

        for nth_job in range(n_jobs):
            nth_process = len(self.search_processes)
            self._add_process(
                nth_process,
                model,
                search_space,
                name,
                n_iter,
                optimizer,
                n_jobs,
                init_para,
                memory,
            )

    def run(self, max_time=None, distribution=None):
        self.search = Search(self.training_data, self.search_processes, self.verbosity)

        start_time = time.time()
        self.search.run(start_time, max_time)

        self.eval_times = self.search.eval_times_dict
        self.iter_times = self.search.iter_times_dict

        self.positions = self.search.positions_dict
        self.scores = self.search.scores_dict
        self.score_best_list = self.search.best_score_list_dict

        self.para_best = self.search.para_best_dict
        self.score_best = self.search.score_best_dict

        self.position_results = self.search.position_results
