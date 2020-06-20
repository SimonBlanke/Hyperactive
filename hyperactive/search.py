# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

import numpy as np
from pathos.multiprocessing import ProcessingPool

from .search_process import SearchProcess


class Search:
    def __init__(self, search_processes, study_para, n_jobs):
        self.study_para = study_para
        self.search_processes = search_processes
        self.n_jobs = n_jobs

        self._n_process_range = range(0, int(n_jobs))

    def run(self):
        self.start_time = time.time()
        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}
        self.pos_list = {}
        self.score_list = {}

        if len(self.search_processes) == 1:
            self._run_job(0)
        else:
            self._run_multiple_jobs()

        return (
            self.results,
            self.pos_list,
            self.score_list,
            self.eval_times,
            self.iter_times,
            self.best_scores,
        )

    def _search_multiprocessing(self):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = ProcessingPool(self.n_jobs)
        self.processlist, _p_list = zip(*pool.map(self._run, self._n_process_range))

        return self.processlist, _p_list

    def _run_job(self, nth_process):
        self.process, _p_ = self._run(nth_process)
        self._get_attributes(_p_)

    def _get_attributes(self, _p_):
        self.results[self.process.obj_func] = self.process._process_results()
        self.eval_times[self.process.obj_func] = self.process.eval_time
        self.iter_times[self.process.obj_func] = self.process.iter_times
        self.best_scores[self.process.obj_func] = self.process.score_best

        if isinstance(_p_, list):
            self.pos_list[self.process.obj_func] = [np.array(p.pos_list) for p in _p_]
            self.score_list[self.process.obj_func] = [
                np.array(p.score_list) for p in _p_
            ]
        else:
            self.pos_list[self.process.obj_func] = [np.array(_p_.pos_list)]
            self.score_list[self.process.obj_func] = [np.array(_p_.score_list)]

    def _run_multiple_jobs(self):
        self.processlist, _p_list = self._search_multiprocessing()

        for _ in range(int(self.n_jobs / 2) + 2):
            print("\n")

        for self.process, _p_ in zip(self.processlist, _p_list):
            self._get_attributes(_p_)

    def _run(self, nth_process):
        process = self.search_processes[nth_process]
        return process.search(nth_process)

    def _time_exceeded(self):
        run_time = time.time() - self.start_time
        return self.study_para.max_time and run_time > self.study_para.max_time

    def _initialize_search(self, study_para, nth_process, _info_):
        study_para._set_random_seed(nth_process)

        self.process = SearchProcess(nth_process, study_para, _info_)
        self._pbar_.init_p_bar(nth_process, self.study_para)
