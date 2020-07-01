# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from importlib import import_module

from .search import Search
from .verbosity import Verbosity


search_process_dict = {
    False: "SearchProcessNoMem",
    "short": "SearchProcessShortMem",
    "long": "SearchProcessLongMem",
}


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
        nth_job,
        objective_function,
        search_space,
        n_iter,
        function_parameter,
        optimizer,
        n_jobs,
        memory,
    ):
        module = import_module(".search_process", "hyperactive")
        search_process_class = getattr(module, search_process_dict[memory])

        search_process_kwargs = {
            "nth_job": nth_job,
            "verb": self.verb,
            "objective_function": objective_function,
            "search_space": search_space,
            "n_iter": n_iter,
            "function_parameter": function_parameter,
            "optimizer": optimizer,
            "n_jobs": n_jobs,
            "memory": memory,
            "hyperactive": self.hyperactive,
        }

        new_search_process = search_process_class(**search_process_kwargs)
        self.search_processes.append(new_search_process)

    def add_search(
        self,
        objective_function,
        search_space,
        n_iter=10,
        function_parameter=None,
        optimizer="RandomSearch",
        n_jobs=1,
        memory="short",
    ):

        for nth_job in range(pro_arg.n_jobs):
            self._add_process(
                nth_job,
                objective_function,
                search_space,
                n_iter,
                function_parameter,
                optimizer,
                n_jobs,
                memory,
            )

        self.search = Search(self.search_processes)

    def run(self, max_time=None, distribution=None):
        if max_time is not None:
            max_time = max_time * 60

        start_time = time.time()

        self.search.run(start_time, max_time)

        self.position_results = self.search.position_results
        self.eval_times = self.search.eval_times
        self.iter_times = self.search.iter_times
        self.best_para = self.search.results
        self.best_score = self.search.results

