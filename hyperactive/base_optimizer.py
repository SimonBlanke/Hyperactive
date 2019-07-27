# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import pickle
import multiprocessing


import tqdm
import numpy as np
from functools import partial

from .base_positioner import BasePositioner
from .config import Config
from .opt_args import Arguments

from .candidate import MlCandidate
from .candidate import DlCandidate


class BaseOptimizer:
    def __init__(self, *args, **kwargs):

        """

        Parameters
        ----------

        search_config: dict
            A dictionary providing the model and hyperparameter search space for the
            optimization process.
        n_iter: int
            The number of iterations the optimizer performs.
        metric: string, optional (default: "accuracy")
            The metric the model is evaluated by.
        n_jobs: int, optional (default: 1)
            The number of searches to run in parallel.
        cv: int, optional (default: 5)
            The number of folds for the cross validation.
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

        self._config_ = Config(*args, **kwargs)
        self._arg_ = Arguments(**kwargs)

        self.search_config = self._config_.search_config
        self.n_iter = self._config_.n_iter

        self._config_.get_model_type()

    def _sort_for_best(self, sort, sort_by):
        """Returns two lists sorted by the second"""
        sort = np.array(sort)
        sort_by = np.array(sort_by)

        index_best = list(sort_by.argsort()[::-1])

        sort_sorted = sort[index_best]
        sort_by_sorted = sort_by[index_best]

        return sort_sorted, sort_by_sorted

    def _init_search(self, nth_process, X, y, init=None):
        """Initializes the search by instantiating the ml- or dl-candidate for each process"""
        self._config_._set_random_seed(nth_process)

        if (
            self._config_.model_type == "sklearn"
            or self._config_.model_type == "xgboost"
        ):

            search_config_key = self._config_._get_sklearn_model(nth_process)
            _cand_ = MlCandidate(
                nth_process,
                self._config_.search_config,
                self._config_.metric,
                self._config_.cv,
                self._config_.warm_start,
                self._config_.memory,
                self._config_.scatter_init,
                search_config_key,
            )

        elif self._config_.model_type == "keras":
            _cand_ = DlCandidate(
                nth_process,
                self._config_.search_config,
                self._config_.metric,
                self._config_.cv,
                self._config_.warm_start,
                self._config_.memory,
                self._config_.scatter_init,
            )

        pos = _cand_._init_._set_start_pos(nth_process, X, y)
        score = _cand_.eval_pos(pos, X, y)
        _cand_.score_best = score
        _cand_.pos_best = pos

        # initialize optimizer specific objects
        if self.initializer:
            _p_ = self.initializer(_cand_, X, y)

        # create progress bar
        if self._config_._show_progress_bar():
            self.p_bar = tqdm.tqdm(**self._config_._tqdm_dict(_cand_))

        return _cand_, _p_

    def _initialize(self, _cand_, positioner=None, pos_para={}):
        if positioner:
            _p_ = positioner(**pos_para)
        else:
            _p_ = BasePositioner(**pos_para)

        _p_.pos_current = _cand_.pos_best
        _p_.score_current = _cand_.score_best

        return _p_

    def _update_pos(self, _cand_, _p_):
        _cand_.pos_best = _p_.pos_new
        _cand_.score_best = _p_.score_new

        _p_.pos_current = _p_.pos_new
        _p_.score_current = _p_.score_new

        return _cand_, _p_

    def search(self, nth_process, X, y):
        _cand_, _p_ = self._init_search(nth_process, X, y)

        for i in range(self._config_.n_iter):
            _cand_ = self._iterate(i, _cand_, _p_, X, y)

            if self._config_._show_progress_bar():
                self.p_bar.update(1)

        return _cand_

    def _search_multiprocessing(self, X, y):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = multiprocessing.Pool(self._config_.n_jobs)
        search = partial(self.search, X=X, y=y)

        _cand_list = pool.map(search, self._config_._n_process_range)

        return _cand_list

    def _run_one_job(self, X, y):
        _cand_ = self.search(0, X, y)

        start_point = _cand_._get_warm_start()
        self.score_best = _cand_.score_best

        para = _cand_._space_.pos2para(_cand_.pos_best)
        self.model_best, _ = _cand_._model_._create_model(para)

        if self._config_.verbosity:
            print("\n", self._config_.metric, self.score_best)
            print("start_point =", start_point)

    def _run_multiple_jobs(self, X, y):
        _cand_list = self._search_multiprocessing(X, y)

        start_point_list = []
        score_best_list = []
        model_best_list = []
        for _cand_ in _cand_list:
            start_point = _cand_._get_warm_start()
            score_best = _cand_.score_best

            para = _cand_._space_.pos2para(_cand_.pos_best)
            model_best = _cand_._model_._create_model(para)

            start_point_list.append(start_point)
            score_best_list.append(score_best)
            model_best_list.append(model_best)

        start_point_sorted, score_best_sorted = self._sort_for_best(
            start_point_list, score_best_list
        )

        model_best_sorted, score_best_sorted = self._sort_for_best(
            model_best_list, score_best_list
        )

        if self._config_.verbosity:
            print("\nList of start points (best first):")
            for start_point, score_best in zip(start_point_sorted, score_best_sorted):
                print("\n", self._config_.metric, score_best)
                print("start_point =", start_point)

        self.score_best = score_best_sorted[0]
        self.model_best = model_best_sorted[0]

    def fit(self, X, y):
        """Public method for starting the search with the training data (X, y)

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]

        Returns
        -------
        None
        """
        X, y = self._config_._check_data(X, y)

        if self._config_.model_type == "keras":
            self._config_.n_jobs = 1

        if self._config_.n_jobs == 1:
            self._run_one_job(X, y)

        else:
            self._run_multiple_jobs(X, y)

        # self.model_best.fit(X, y)

    def predict(self, X_test):
        """Returns the prediction of X_test after a model was searched by `fit`

        Parameters
        ----------
        X_test : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        (unnamed array) : array-like, shape = [n_samples] or [n_samples, n_outputs]
        """
        return self.model_best.predict(X_test)

    def score(self, X_test, y_test):
        """Returns the score calculated from the prediction of X_test and the true values from y_test

        Parameters
        ----------
        X_test : array-like or sparse matrix of shape = [n_samples, n_features]

        y_test : array-like, shape = [n_samples] or [n_samples, n_outputs]

        Returns
        -------
        (unnamed float) : float
        """
        if (
            self._config_.model_type == "sklearn"
            or self._config_.model_type == "xgboost"
        ):
            return self.model_best.score(X_test, y_test)
        elif self._config_.model_type == "keras":
            return self.model_best.evaluate(X_test, y_test)[1]

    def export(self, filename):
        """Exports the best model, that was found by the optimizer during `fit`

        Parameters
        ----------
        filename : string or path

        Returns
        -------
        None
        """
        if self.model_best:
            pickle.dump(self.model_best, open(filename, "wb"))
