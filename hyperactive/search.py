# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from multiprocessing import Pool
from importlib import import_module


class SearchBase:
    def __init__(self, function_parameter, search_processes):
        self.function_parameter = function_parameter
        self.search_processes = search_processes
        self.n_processes = len(search_processes)
        self._n_process_range = range(0, self.n_processes)

        self.obj_functions = self._uniques_obj_func(search_processes)

        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}
        self.pos_list = {}
        self.score_list = {}
        self.position_results = {}

    def _uniques_obj_func(self, search_processes):
        self.obj_func_list = []
        for process in search_processes:
            self.obj_func_list.append(process.objective_function)

        return set(self.obj_func_list)

    def _get_results(self):
        position_results_dict = {}

        eval_times_dict = {}
        iter_times_dict = {}

        score_best_dict = {}
        para_best_dict = {}

        score_best = -np.inf
        para_best = None
        for objective_function in self.obj_func_set:
            for process in self.search_processes:
                if objective_function != process.objective_function:
                    continue

                if score_best < process.cand.score_best:
                    score_best = process.cand.score_best
                    para_best = process.cand.para_best

            score_best_dict[objective_function] = score_best
            para_best_dict[objective_function] = para_best

    def _print_best_para(self):
        for _ in range(int(self.n_processes / 2) + 2):
            print("\n")  # make room in cmd for prints
        for process in self.search_processes:
            process.print_best_para()

    def _run_job(self, nth_process):
        self.process = self.search_processes[nth_process]
        return self.process.search(self.start_time, self.max_time, nth_process)

    def _run_multiple_jobs(self):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = Pool(self.n_processes)
        results_list = pool.map(self._run_job, self._n_process_range)

        return results_list

    def _memory_dict2dataframe(self, results_dict):
        memory_dict = results_dict["memory"]
        tuple_list = list(memory_dict.keys())
        result_list = list(memory_dict.values())

        results_df = pd.DataFrame(result_list)
        np_pos = np.array(tuple_list)

        columns = list(results_dict["search_space"].keys())
        columns = [col + ".index" for col in columns]
        pd_pos = pd.DataFrame(np_pos, columns=columns)

        results = pd.concat([pd_pos, results_df], axis=1)

        return results

    def _run(self, start_time, max_time):
        self.start_time = start_time
        self.max_time = max_time

        if len(self.search_processes) == 1:
            results_list = [self._run_job(0)]
        else:
            results_list = self._run_multiple_jobs()

        return results_list

        """
        for result in results_list:
            # print("\n result \n", result)

            self._memory_dict2dataframe(result)

            # self._store_memory(result["memory"])
            # self._print_best_para()
        """


class Search(SearchBase):
    def __init__(self, function_parameter, search_processes):
        super().__init__(function_parameter, search_processes)

    def run(self, start_time, max_time):
        self._run(start_time, max_time)


class SearchLongTermMemory(Search):
    def __init__(self, function_parameter, search_processes):
        super().__init__(function_parameter, search_processes)
        self._load_memory()

    def _load_memory(self):
        for process in self.search_processes:
            process.cand.memory_dict = process.res.load_long_term_memory()

    def _save_memory(self):
        for process in self.search_processes:
            process.res.save_long_term_memory()

    def run(self, start_time, max_time):
        self._run(start_time, max_time)
        self._save_memory()

