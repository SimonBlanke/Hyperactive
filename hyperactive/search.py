# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from multiprocessing import Pool


class Search:
    def __init__(self, search_processes):
        self.search_processes = search_processes
        self.n_processes = len(search_processes)
        self._n_process_range = range(0, self.n_processes)

    def run(self, start_time, max_time):
        self.start_time = start_time
        self.max_time = max_time

        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}
        self.pos_list = {}
        self.score_list = {}
        self.position_results = {}

        if len(self.search_processes) == 1:
            results_list = [self._run_job(0)]
        else:
            results_list = self._run_multiple_jobs()

        for result in results_list:

            self._store_memory(result["memory"])
            self._print_best_para()

    def _store_memory(self, memory):
        for process in self.search_processes:
            process.store_memory(memory)

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
        return zip(*pool.map(self._run_job, self._n_process_range))

