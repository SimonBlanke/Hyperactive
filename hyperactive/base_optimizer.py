# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numpy as np
import multiprocessing

from .base_positioner import BasePositioner
from .candidate import Candidate
from .verbosity import set_verbosity


class BaseOptimizer:
    def __init__(self, _main_args_, _opt_args_):
        self._main_args_ = _main_args_
        self._opt_args_ = _opt_args_

        self._info_, _pbar_ = set_verbosity(_main_args_.verbosity)
        self._pbar_ = _pbar_()

    def search(self, nth_process=0, rayInit=False):
        self.start_time = time.time()
        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}
        self.pos_list = {}
        self.score_list = {}

        if rayInit:
            self._run_job(nth_process)
        elif self._main_args_.n_jobs == 1:
            self._run_job(nth_process)
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
        pool = multiprocessing.Pool(self._main_args_.n_jobs)
        _cand_list, _p_list = zip(
            *pool.map(self._search, self._main_args_._n_process_range)
        )

        return _cand_list, _p_list

    def _run_job(self, nth_process):
        _cand_, _p_ = self._search(nth_process)
        self._get_attributes(_cand_, _p_)

    def _get_attributes(self, _cand_, _p_):
        self.results[_cand_.func_] = _cand_._process_results(self._opt_args_)
        self.eval_times[_cand_.func_] = _cand_.eval_time
        self.iter_times[_cand_.func_] = _cand_.iter_times
        self.best_scores[_cand_.func_] = _cand_.score_best

        if isinstance(_p_, list):
            self.pos_list[_cand_.func_] = [np.array(p.pos_list) for p in _p_]
            self.score_list[_cand_.func_] = [np.array(p.score_list) for p in _p_]
        else:
            self.pos_list[_cand_.func_] = [np.array(_p_.pos_list)]
            self.score_list[_cand_.func_] = [np.array(_p_.score_list)]

    def _run_multiple_jobs(self):
        _cand_list, _p_list = self._search_multiprocessing()

        for _ in range(int(self._main_args_.n_jobs / 2) + 2):
            print("\n")

        for _cand_, _p_ in zip(_cand_list, _p_list):
            self._get_attributes(_cand_, _p_)

    def _search(self, nth_process):
        _cand_ = self._initialize_search(self._main_args_, nth_process, self._info_)

        for i in range(1, self._main_args_.n_iter):
            c_time = time.time()

            _cand_.i = i
            _cand_ = self._iterate(i, _cand_)

            if self._time_exceeded():
                break

            _cand_.iter_times.append(time.time() - c_time)

        _p_ = self._finish_search()
        return _cand_, _p_

    def _time_exceeded(self):
        run_time = time.time() - self.start_time
        return self._main_args_.max_time and run_time > self._main_args_.max_time

    def _initialize_search(self, _main_args_, nth_process, _info_):
        _main_args_._set_random_seed(nth_process)

        _cand_ = Candidate(nth_process, _main_args_, _info_)
        self._pbar_.init_p_bar(nth_process, self._main_args_)

        self._init_iteration(_cand_)

        return _cand_

    def _init_base_positioner(self, _cand_, positioner=None):
        if positioner:
            _p_ = positioner(**self._opt_args_.kwargs_opt)
        else:
            _p_ = BasePositioner(**self._opt_args_.kwargs_opt)

        _p_.pos_new = _cand_.pos_best
        _p_.score_new = _cand_.score_best

        return _p_

    def _update_pos(self, _cand_, _p_):
        if _p_.score_new > _p_.score_best:
            _p_.pos_best = _p_.pos_new
            _p_.score_best = _p_.score_new

        if _p_.score_new > _cand_.score_best:
            _p_.pos_current = _p_.pos_new
            _p_.score_current = _p_.score_new

            _cand_.pos_best = _p_.pos_new
            _cand_.score_best = _p_.score_new

            self._pbar_.best_since_iter = _cand_.i

    def _optimizer_eval(self, _cand_, _p_):
        _p_.score_new = _cand_.eval_pos(_p_.pos_new)
        self._pbar_.update_p_bar(1, _cand_)
