# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time
import pickle
import random
import multiprocessing

import scipy
import numpy as np
import xgboost as xgb

from importlib import import_module
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from functools import partial


class BaseOptimizer(object):
    def __init__(
        self,
        search_space,
        n_iter,
        scoring="accuracy",
        tabu_memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
    ):

        self.search_space = search_space
        self.n_iter = n_iter
        self.scoring = scoring
        self.tabu_memory = tabu_memory
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbosity = verbosity
        self.random_state = random_state
        self.start_points = start_points

        self.X_train = None
        self.y_train = None
        self.init_search_dict = None

        self._set_n_jobs()

        # model_str = random.choice(list(self.search_dict.keys()))
        self.hyperpara_search_dict = search_space[list(search_space.keys())[0]]

        # self._set_random_seed()
        self._n_process_range = range(0, self.n_jobs)
        self._limit_pos()

    def _set_n_jobs(self):
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores
        if self.n_jobs > self.n_iter:
            self.n_iter = self.n_jobs

    def _set_random_seed(self, thread=0):
        if self.random_state:
            random.seed(self.random_state + thread)
            np.random.seed(self.random_state + thread)
            scipy.random.seed(self.random_state + thread)

    def _set_n_steps(self, n_process):
        n_steps = int(self.n_iter / self.n_jobs)
        remain = self.n_iter % self.n_jobs

        if n_process < remain:
            n_steps += 1

        return n_steps

    def _get_dim_SearchSpace(self):
        return len(self.hyperpara_search_dict)

    def _limit_pos(self):
        max_pos_list = []
        for values in list(self.hyperpara_search_dict.values()):
            max_pos_list.append(len(values) - 1)

        self.max_pos_list = np.array(max_pos_list)

    def _find_best_model(self, models, scores):
        N_best_models = 1

        scores = np.array(scores)
        index_best_scores = scores.argsort()[-N_best_models:][::-1]

        best_score = scores[index_best_scores]
        best_model = models[index_best_scores[0]]

        return best_model, best_score

    def _search(self):
        pass

    def _search_multiprocessing(self, X_train, y_train):
        pool = multiprocessing.Pool(self.n_jobs)

        _search = partial(self._search, X_train=X_train, y_train=y_train)

        models, scores, hyperpara_dict, train_time = zip(
            *pool.map(_search, self._n_process_range)
        )

        self.model_list = models
        self.score_list = scores
        self.hyperpara_dict = hyperpara_dict
        self.train_time = train_time

        return models, scores

    def fit(self, X_train, y_train):
        models, scores = self._search_multiprocessing(X_train, y_train)

        self.best_model, best_score = self._find_best_model(models, scores)
        self.best_model.fit(X_train, y_train)

        if self.verbosity:
            print("Best score:", *best_score)
            print("Best model:", self.best_model)

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def export(self, filename):
        if self.best_model:
            pickle.dump(self.best_model, open(filename, "wb"))


class MachineLearner:
    def __init__(self, search_space, scoring, cv):
        self.search_space = search_space
        self.scoring = scoring
        self.cv = cv

        self.model_key = list(self.search_space.keys())[0]

        self._check_model_key()

    def _check_model_key(self):
        if "sklearn" and "xgboost" not in self.model_key:
            raise ValueError("No sklearn model in search_dict found")

    def _get_model(self, model):
        sklearn, submod_func = model.rsplit(".", 1)
        module = import_module(sklearn)
        model = getattr(module, submod_func)

        return model

    def _create_sklearn_model(self, model, hyperpara_dict):
        return model(**hyperpara_dict)

    def train_model(self, hyperpara_dict, X_train, y_train):
        model = self._get_model(self.model_key)
        sklearn_model = self._create_sklearn_model(model, hyperpara_dict)

        time_temp = time.perf_counter()
        scores = cross_val_score(
            sklearn_model, X_train, y_train, scoring=self.scoring, cv=self.cv
        )
        train_time = (time.perf_counter() - time_temp) / self.cv

        return scores.mean(), train_time, sklearn_model


class SearchSpace:
    def __init__(self, start_points, search_space):
        self.start_points = start_points
        self.hyperpara_search_dict = search_space[list(search_space.keys())[0]]

    def init_eval(self, n_process):
        hyperpara_indices = None
        if self.start_points:
            for key in self.start_points.keys():
                if key == n_process:
                    hyperpara_indices = self.set_start_position(n_process)

        if not hyperpara_indices:
            hyperpara_indices = self.get_random_position()

        return hyperpara_indices

    def set_start_position(self, n_process):
        pos_dict = {}

        for hyperpara_name in self.hyperpara_search_dict.keys():
            search_position = self.hyperpara_search_dict[hyperpara_name].index(
                self.start_points[n_process][hyperpara_name]
            )

            pos_dict[hyperpara_name] = search_position

        return pos_dict

    def get_random_position(self):
        """
        get a random N-Dim position in search space and return:
        N indices of N-Dim position (dict)
        """
        pos_dict = {}

        for hyperpara_name in self.hyperpara_search_dict.keys():
            n_hyperpara_values = len(self.hyperpara_search_dict[hyperpara_name])
            search_position = random.randint(0, n_hyperpara_values - 1)

            pos_dict[hyperpara_name] = search_position

        return pos_dict

    def pos_dict2values_dict(self, pos_dict):
        values_dict = {}

        for hyperpara_name in pos_dict.keys():
            pos = pos_dict[hyperpara_name]
            values_dict[hyperpara_name] = list(
                self.hyperpara_search_dict[hyperpara_name]
            )[pos]

        return values_dict

    def pos_dict2np_array(self, pos_dict):
        return np.array(list(pos_dict.values()))

    def pos_np2values_dict(self, np_array):
        if len(self.hyperpara_search_dict.keys()) == np_array.size:
            values_dict = {}
            for i, key in enumerate(self.hyperpara_search_dict.keys()):
                pos = int(np_array[i])
                values_dict[key] = list(self.hyperpara_search_dict[key])[pos]

            return values_dict
        else:
            raise ValueError("hyperpara_search_dict and np_array have different size")
