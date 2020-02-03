# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numpy as np
import multiprocessing

from .base_positioner import BasePositioner
from .verbosity import set_verbosity
from .candidate import Candidate


class BaseOptimizer:
    def __init__(self, _main_args_, _opt_args_):
        self._main_args_ = _main_args_
        self._opt_args_ = _opt_args_

        self._info_, _pbar_ = set_verbosity(_main_args_.verbosity)
        self._pbar_ = _pbar_()

        self.pos_list = []
        self.score_list = []

    def _init_base_positioner(self, _cand_, positioner=None):
        if positioner:
            _p_ = positioner(**self._opt_args_.kwargs_opt)
        else:
            _p_ = BasePositioner(**self._opt_args_.kwargs_opt)

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_

    def _update_pos(self, _cand_, _p_):
        _cand_.pos_best = _p_.pos_new
        _cand_.score_best = _p_.score_new

        _p_.pos_current = _p_.pos_new
        _p_.score_current = _p_.score_new

        self._pbar_.best_since_iter = _cand_.i

        return _cand_, _p_

    def _initialize_search(self, _main_args_, nth_process):
        _main_args_._set_random_seed(nth_process)
        _cand_ = Candidate(nth_process, _main_args_, self._info_)
        self._pbar_.init_p_bar(_cand_, self._main_args_)

        _p_ = self._init_opt_positioner(_cand_)
        self._pbar_.update_p_bar(1, _cand_)

        return _cand_, _p_

    def _search(self, nth_process):
        _cand_, _p_ = self._initialize_search(self._main_args_, nth_process)

        for i in range(self._main_args_.n_iter - 1):
            c_time = time.time()

            _cand_.i = i
            _cand_ = self._iterate(i, _cand_, _p_)
            self._pbar_.update_p_bar(1, _cand_)

            run_time = time.time() - self.start_time
            if self._main_args_.max_time and run_time > self._main_args_.max_time:
                break

            if self._main_args_.get_search_path:
                self._monitor_search_path(_p_)

            _cand_.iter_times.append(time.time() - c_time)

        self._pbar_.close_p_bar()
        return _cand_

    def _monitor_search_path(self, _p_):
        pos_list = []
        score_list = []
        if isinstance(_p_, list):
            for p in _p_:
                pos_list.append(p.pos_new)
                score_list.append(p.score_new)

                pos_list_ = np.array(pos_list)
                score_list_ = np.array(score_list)

            self.pos_list.append(pos_list_)
            self.score_list.append(score_list_)
        else:
            pos_list.append(_p_.pos_new)
            score_list.append(_p_.score_new)

            pos_list_ = np.array(pos_list)
            score_list_ = np.array(score_list)

            self.pos_list.append(pos_list_)
            self.score_list.append(score_list_)

    def _search_multiprocessing(self):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = multiprocessing.Pool(self._main_args_.n_jobs)
        _cand_list = pool.map(self._search, self._main_args_._n_process_range)

        return _cand_list

    def _run_job(self, nth_process):
        _cand_ = self._search(nth_process)
        self.results[_cand_.func_] = _cand_._process_results(self._opt_args_)
        self.eval_times[_cand_.func_] = _cand_.eval_time
        self.iter_times[_cand_.func_] = _cand_.iter_times
        self.best_scores[_cand_.func_] = _cand_.score_best

    def _run_multiple_jobs(self):
        _cand_list = self._search_multiprocessing()

        for _ in range(int(self._main_args_.n_jobs / 2) + 2):
            print("\n")

        for _cand_ in _cand_list:
            self.results[_cand_.func_] = _cand_._process_results(self._opt_args_)
            self.eval_times[_cand_.func_] = _cand_.eval_time
            self.iter_times[_cand_.func_] = _cand_.iter_times
            self.best_scores[_cand_.func_] = _cand_.score_best

    def search(self, nth_process=0, rayInit=False):
        self.start_time = time.time()
        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}

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
