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
            self._run_job(0)
        else:
            self._run_multiple_jobs()

    def _search_multiprocessing(self):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = Pool(self.n_processes)
        _p_list = zip(*pool.map(self._run, self._n_process_range))

        return _p_list

    def _run_job(self, nth_process):
        _p_ = self._run(nth_process)
        self._get_attributes(_p_)

    def _get_attributes(self, _p_):
        # self.results[self.process.obj_func] = self.process._process_results()
        self.process._process_results()

        # self.eval_times[self.process.obj_func] = self.process.eval_time
        # self.iter_times[self.process.obj_func] = self.process.iter_times
        # self.best_score[self.process.obj_func] = self.process.score_best
        # self.best_para[self.process.obj_func] = self.process.best_para
        self.position_results[self.process.obj_func] = self.process.position_results

        if isinstance(_p_, list):
            self.pos_list[self.process.obj_func] = [np.array(p.pos_list) for p in _p_]
            self.score_list[self.process.obj_func] = [
                np.array(p.score_list) for p in _p_
            ]
        else:
            self.pos_list[self.process.obj_func] = [np.array(_p_.pos_list)]
            self.score_list[self.process.obj_func] = [np.array(_p_.score_list)]

    def _run_multiple_jobs(self):
        _p_list = self._search_multiprocessing()
        for _ in range(int(self.n_processes / 2) + 2):
            print("\n")

        """
        for self.process, _p_ in zip(self.processlist, _p_list):
            self._get_attributes(_p_)
        """

    def _run(self, nth_process):
        self.process = self.search_processes[nth_process]
        return self.process.search(self.start_time, self.max_time, nth_process)
