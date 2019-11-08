# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numpy as np
import multiprocessing

from functools import partial

from .base_positioner import BasePositioner
from .verb import VerbosityLVL0, VerbosityLVL1, VerbosityLVL2
from .util import init_candidate, init_eval
from .candidate import Candidate
from meta_learn import HyperactiveWrapper


class BaseOptimizer:
    def __init__(self, _core_, _arg_):

        """

        Parameters
        ----------

        search_config: dict
            A dictionary providing the model and hyperparameter search space for the
            optimization process.
        n_iter: int
            The number of iterations the optimizer performs.
        n_jobs: int, optional (default: 1)
            The number of searches to run in parallel.
        verbosity: int, optional (default: 1)
            Verbosity level. 1 prints out warm_start points and their scores.
        random_state: int, optional (default: None)
            Sets the random seed.
        warm_start: dict, optional (default: False)
            Dictionary that definies a start point for the optimizer.
        memory: bool, optional (default: True)
            A memory, that saves the evaluation during the optimization to save time when
            optimizer returns to position.
        scatter_init: int, optional (default: False)
            Defines the number n of random positions that should be evaluated with 1/n the
            training data, to find a better initial position.

        Returns
        -------
        None

        """

        self._core_ = _core_
        self._arg_ = _arg_
        self._meta_ = None

        self.search_config = self._core_.search_config
        self.n_iter = self._core_.n_iter

        if self._core_.memory == "long":
            self._meta_ = HyperactiveWrapper(self._core_.search_config)

        verbs = [VerbosityLVL0, VerbosityLVL1, VerbosityLVL2]
        self._verb_ = verbs[_core_.verbosity]()

        self.pos_list = []
        self.score_list = []

    def _init_base_positioner(self, _cand_, positioner=None):
        if positioner:
            _p_ = positioner(**self._arg_.kwargs_opt)
        else:
            _p_ = BasePositioner(**self._arg_.kwargs_opt)

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_

    def _update_pos(self, _cand_, _p_):
        _cand_.pos_best = _p_.pos_new
        _cand_.score_best = _p_.score_new

        _p_.pos_current = _p_.pos_new
        _p_.score_current = _p_.score_new

        self._verb_.best_since_iter = _cand_.i

        return _cand_, _p_

    def _initialize_search(self, _core_, nth_process, X, y):
        _cand_ = init_candidate(_core_, nth_process, Candidate)
        _cand_ = init_eval(_cand_, nth_process, X, y)
        _p_ = self._init_opt_positioner(_cand_, X, y)
        self._verb_.init_p_bar(_cand_, self._core_)

        if self._meta_:
            meta_data = self._meta_.get_func_metadata(_cand_)

            # self._meta_.retrain(_cand_)
            # para, score = self._meta_.search(X, y, _cand_)

            _cand_._space_.load_memory(*meta_data)

        return _core_, _cand_, _p_

    def _finish_search(self, _core_, _cand_, X, y):
        _cand_.eval_pos(_cand_.pos_best, X, y, force_eval=True)
        self.eval_time = _cand_.eval_time_sum
        self._verb_.close_p_bar()

        return _cand_

    def search(self, nth_process, X, y):
        self._core_, _cand_, _p_ = self._initialize_search(
            self._core_, nth_process, X, y
        )

        for i in range(self._core_.n_iter):
            _cand_.i = i
            _cand_ = self._iterate(i, _cand_, _p_, X, y)
            self._verb_.update_p_bar(1, _cand_)

            run_time = time.time() - self.start_time
            if self._core_.max_time and run_time > self._core_.max_time:
                break

            # get_search_path
            if self._core_.get_search_path:
                self._monitor_search_path(_p_)

        _cand_ = self._finish_search(self._core_, _cand_, X, y)

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

    def _process_results(self, X, y, _cand_):
        start_point = self._verb_.print_start_point(_cand_)
        self.results_params[_cand_.func_] = start_point
        self.results_models[_cand_.func_] = _cand_.model_best

        if self._core_.memory == "long":
            self._meta_.collect(X, y, _cand_)

    def _search_multiprocessing(self, X, y):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = multiprocessing.Pool(self._core_.n_jobs)
        search = partial(self.search, X=X, y=y)

        _cand_list = pool.map(search, self._core_._n_process_range)

        return _cand_list

    def _run_one_job(self, X, y):
        _cand_ = self.search(0, X, y)
        self._process_results(X, y, _cand_)

    def _run_multiple_jobs(self, X, y):
        _cand_list = self._search_multiprocessing(X, y)

        for _ in range(int(self._core_.n_jobs / 2)):
            print("\n")

        for _cand_ in _cand_list:
            self._process_results(X, y, _cand_)

    def _fit(self, X, y):
        """Public method for starting the search with the training data (X, y)

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]

        Returns
        -------
        None
        """
        self.start_time = time.time()
        self.results_params = {}
        self.results_models = {}

        if self._core_.n_jobs == 1:
            self._run_one_job(X, y)
        else:
            self._run_multiple_jobs(X, y)
