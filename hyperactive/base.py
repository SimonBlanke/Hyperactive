# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import pickle
import multiprocessing
import random

import scipy
import numpy as np
import pandas as pd

from importlib import import_module
from functools import partial

from .candidate import MlCandidate
from .candidate import DlCandidate


class BaseOptimizer(object):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        memory=True,
        scatter_init=False,
    ):
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

        self.search_config = search_config
        self.n_iter = n_iter
        self.metric = metric
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbosity = verbosity
        self.random_state = random_state
        self.warm_start = warm_start
        self.memory = memory
        self.scatter_init = scatter_init

        self.X_train = None
        self.y_train = None
        self.model_type = None

        self._get_model_type()

        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        self._set_n_jobs()

        self._n_process_range = range(0, int(self.n_jobs))

    def _tqdm_dict(self, _cand_):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""
        return {
            "iterable": range(self.n_steps),
            # "desc": str(self.model_str),
            "position": _cand_.nth_process,
            "leave": False,
        }

    def _set_random_seed(self, thread=0):
        """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
        if isinstance(self.random_state, int) and self.random_state is not False:
            random.seed(self.random_state + thread)
            np.random.seed(self.random_state + thread)
            scipy.random.seed(self.random_state + thread)

    def _is_all_same(self, list):
        """Checks if model names in search_config are consistent"""
        if len(set(list)) == 1:
            return True
        else:
            return False

    def _get_model_type(self):
        """extracts the model type from the search_config (important for search space construction)"""
        model_type_list = []

        for model_type_key in self.search_config.keys():
            if "sklearn" in model_type_key:
                model_type_list.append("sklearn")
            elif "xgboost" in model_type_key:
                model_type_list.append("xgboost")
            elif "keras" in model_type_key:
                model_type_list.append("keras")
            elif "torch" in model_type_key:
                model_type_list.append("torch")
            else:
                raise Exception("\n No valid model string in search_config")

        if self._is_all_same(model_type_list):
            self.model_type = model_type_list[0]
        else:
            raise Exception("\n Model strings in search_config keys are inconsistent")

    def _get_sklearn_model(self, nth_process):
        """Gets a model_key from the model_list for each thread"""
        if self.n_models > self.n_jobs:
            diff = self.n_models - self.n_jobs

            if nth_process == 0:
                print(
                    "\nNot enough jobs to process models. The last",
                    diff,
                    "model(s) will not be processed",
                )
            model_key = self.model_list[nth_process]
        elif nth_process < self.n_models:
            model_key = self.model_list[nth_process]
        else:
            model_key = random.choice(self.model_list)

        return model_key

    def _set_n_jobs(self):
        """Sets the number of jobs to run in parallel"""
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores
        if self.n_jobs > self.n_iter:
            self.n_iter = self.n_jobs

    def _set_n_steps(self, nth_process):
        """Calculates the number of steps each process has to do"""
        n_steps = int(self.n_iter / self.n_jobs)
        remain = self.n_iter % self.n_jobs

        if nth_process < remain:
            n_steps += 1

        return n_steps

    def _sort_for_best(self, sort, sort_by):
        """Returns two lists sorted by the second"""
        sort = np.array(sort)
        sort_by = np.array(sort_by)

        index_best = list(sort_by.argsort()[::-1])

        sort_sorted = sort[index_best]
        sort_by_sorted = sort_by[index_best]

        return sort_sorted, sort_by_sorted

    def _init_search(self, nth_process, X, y):
        """Initializes the search by instantiating the ml- or dl-candidate for each process"""
        self._set_random_seed(nth_process)
        self.n_steps = self._set_n_steps(nth_process)

        if self.model_type == "sklearn" or self.model_type == "xgboost":

            search_config_key = self._get_sklearn_model(nth_process)
            _cand_ = MlCandidate(
                nth_process,
                self.search_config,
                self.metric,
                self.cv,
                self.warm_start,
                self.memory,
                self.scatter_init,
                search_config_key,
            )

        elif self.model_type == "keras":
            _cand_ = DlCandidate(
                nth_process,
                self.search_config,
                self.metric,
                self.cv,
                self.warm_start,
                self.memory,
                self.scatter_init,
            )

        _cand_.pos = _cand_._init_._set_start_pos(nth_process, X, y)

        return _cand_

    def _get_model(self, model):
        """Imports model from search_config key and returns usable model-class"""
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model

    def _search_multiprocessing(self, X, y):
        """Wrapper for the parallel search. Passes integer that corresponds to process number"""
        pool = multiprocessing.Pool(self.n_jobs)
        search = partial(self.search, X=X, y=y)

        _cand_list = pool.map(search, self._n_process_range)

        return _cand_list

    def _check_data(self, X, y):
        """Checks if data is pandas Dataframe and converts to numpy array if necessary"""
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(X, pd.core.frame.DataFrame):
            y = y.values

        return X, y

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
        X, y = self._check_data(X, y)

        if self.model_type == "keras":
            self.n_jobs = 1

        if self.n_jobs == 1:
            _cand_ = self.search(0, X, y)

            start_point = _cand_._get_warm_start()
            self.score_best = _cand_.score_best
            self.model_best = _cand_.model_best

            if self.verbosity:
                print("\n", self.metric, self.score_best)
                print("start_point =", start_point)

        else:
            _cand_list = self._search_multiprocessing(X, y)

            start_point_list = []
            score_best_list = []
            model_best_list = []
            for _cand_ in _cand_list:
                start_point = _cand_._get_warm_start()
                score_best = _cand_.score_best
                model_best = _cand_.model_best

                start_point_list.append(start_point)
                score_best_list.append(score_best)
                model_best_list.append(model_best)

            start_point_sorted, score_best_sorted = self._sort_for_best(
                start_point_list, score_best_list
            )

            model_best_sorted, score_best_sorted = self._sort_for_best(
                model_best_list, score_best_list
            )

            if self.verbosity:
                print("\nList of start points (best first):")
                for start_point, score_best in zip(
                    start_point_sorted, score_best_sorted
                ):
                    print("\n", self.metric, score_best)
                    print("start_point =", start_point)

            self.score_best = score_best_sorted[0]
            self.model_best = model_best_sorted[0]

        self.model_best.fit(X, y)

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
        if self.model_type == "sklearn" or self.model_type == "xgboost":
            return self.model_best.score(X_test, y_test)
        elif self.model_type == "keras":
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
