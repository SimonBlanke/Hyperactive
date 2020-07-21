# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from importlib import import_module

import multiprocessing
from .verbosity import Verbosity

from .checks import check_args

search_process_dict = {
    False: "SearchProcessNoMem",
    "short": "SearchProcessShortMem",
    "long": "SearchProcessLongMem",
}

search_dict = {
    False: "Search",
    "short": "Search",
    "long": "SearchLongTermMemory",
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


class Optimizer:
    def __init__(
        self,
        random_state=None,
        verbosity=3,
        warnings=False,
        ext_warnings=False,
        hyperactive=False,
    ):
        self.verb = Verbosity(verbosity, warnings)
        self.random_state = random_state
        self.hyperactive = hyperactive
        self.search_processes = []

    def _add_process(
        self,
        nth_process,
        objective_function,
        search_space,
        name,
        n_iter,
        function_parameter,
        optimizer,
        n_jobs,
        init_para,
        memory,
    ):
        search_process_kwargs = {
            "nth_process": nth_process,
            "verb": self.verb,
            "objective_function": objective_function,
            "search_space": search_space,
            "search_name": name,
            "n_iter": n_iter,
            "function_parameter": function_parameter,
            "optimizer": optimizer,
            "n_jobs": n_jobs,
            "init_para": init_para,
            "memory": memory,
            "hyperactive": self.hyperactive,
            "random_state": self.random_state,
        }
        SearchProcess = get_class(".search_process", search_process_dict[memory])
        new_search_process = SearchProcess(**search_process_kwargs)
        self.search_processes.append(new_search_process)

    def add_search(
        self,
        objective_function,
        search_space,
        name=None,
        n_iter=10,
        function_parameter=None,
        optimizer="RandomSearch",
        n_jobs=1,
        init_para=[],
        memory="short",
    ):

        check_args(
            objective_function,
            search_space,
            n_iter,
            function_parameter,
            optimizer,
            n_jobs,
            init_para,
            memory,
        )

        n_jobs = set_n_jobs(n_jobs)

        for nth_job in range(n_jobs):
            nth_process = len(self.search_processes)
            self._add_process(
                nth_process,
                objective_function,
                search_space,
                name,
                n_iter,
                function_parameter,
                optimizer,
                n_jobs,
                init_para,
                memory,
            )

        Search = get_class(".search", search_dict[memory])
        self.search = Search(function_parameter, self.search_processes)

    def run(self, max_time=None, distribution=None):
        if max_time is not None:
            max_time = max_time * 60

        start_time = time.time()

        self.search.run(start_time, max_time)

        # self.position_results = self.search.position_results
        self.eval_times = self.search.eval_times_dict
        self.iter_times = self.search.iter_times_dict
        self.best_para = self.search.para_best_dict
        self.best_score = self.search.score_best_dict

