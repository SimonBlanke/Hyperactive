# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from multiprocessing import Pool
from importlib import import_module


class Search:
    def __init__(self, function_parameter, search_processes, verbosity):
        self.function_parameter = function_parameter
        self.search_processes = search_processes
        self.verbosity = verbosity

        self.n_processes = len(search_processes)
        self._n_process_range = range(0, self.n_processes)

        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}
        self.pos_list = {}
        self.score_list = {}
        self.position_results = {}

    def _get_results(self, results_list):
        position_results_dict = {}

        self.eval_times_dict = {}
        self.iter_times_dict = {}

        self.positions_dict = {}
        self.scores_dict = {}
        self.best_score_list_dict = {}

        self.para_best_dict = {}
        self.score_best_dict = {}
        self.memory_dict_new = {}

        for results in results_list:
            search_name = results.search_name

            self.eval_times_dict[search_name] = results.eval_times
            self.iter_times_dict[search_name] = results.iter_times

            self.positions_dict[search_name] = results.pos_list
            self.scores_dict[search_name] = results.score_list
            self.best_score_list_dict[search_name] = results.best_score_list

            self.para_best_dict[search_name] = results.para_best
            self.score_best_dict[search_name] = results.score_best
            self.memory_dict_new[search_name] = results.memory_dict_new
            self.position_results[search_name] = self._memory_dict2dataframe(
                results.memory_dict_new, results.search_space
            )

            if self.verbosity == 0:
                continue

            print(
                "Process",
                results.nth_process,
                "->",
                results.model.__name__,
                "search results:",
            )
            print("best parameter =", results.para_best)
            print("best score     =", results.score_best, "\n")

    def _run_job(self, nth_process):
        self.process = self.search_processes[nth_process]
        return self.process.search(self.start_time, self.max_time, nth_process)

    def _run_multiple_jobs(self):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = Pool(self.n_processes)
        results_list = pool.map(self._run_job, self._n_process_range)

        for _ in range(int(self.n_processes / 2) + 2):
            print("\n")  # make room in cmd for prints

        return results_list

    def _memory_dict2dataframe(self, memory_dict, search_space):
        columns = list(search_space.keys())

        if not bool(memory_dict):
            return pd.DataFrame([], columns=columns)

        pos_tuple_list = list(memory_dict.keys())
        result_list = list(memory_dict.values())

        results_df = pd.DataFrame(result_list)
        np_pos = np.array(pos_tuple_list)

        pd_pos = pd.DataFrame(np_pos, columns=columns)
        dataframe = pd.concat([pd_pos, results_df], axis=1)

        return dataframe

    def _run(self, start_time, max_time):
        self.start_time = start_time
        self.max_time = max_time

        if len(self.search_processes) == 1:
            results_list = [self._run_job(0)]
        else:
            results_list = self._run_multiple_jobs()

        self._get_results(results_list)
        self._save_memory(results_list)

    def run(self, start_time, max_time):
        self._run(start_time, max_time)

    def _save_memory(self, results):
        for result in results:
            if result.memory == "long":
                result.save_long_term_memory()

