# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import pickle
import random
import multiprocessing

import scipy
import numpy as np


from sklearn.metrics import accuracy_score
from functools import partial


class BaseOptimizer(object):
    def __init__(
        self,
        search_config,
        n_iter,
        scoring="accuracy",
        tabu_memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
    ):

        self.search_config = search_config
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
        self.search_space = search_config[list(search_config.keys())[0]]

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
        return len(self.search_space)

    def _limit_pos(self):
        max_pos_list = []
        for values in list(self.search_space.values()):
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

    def _search_normalprocessing(self, X_train, y_train):
        best_model, best_score, best_hyperpara_dict, best_train_time = self._search(
            0, X_train, y_train
        )

        return best_model, best_score

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
        if self.n_jobs == 1:
            self.best_model, best_score = self._search_normalprocessing(
                X_train, y_train
            )
            if self.verbosity:
                print("Best score:", best_score)
                print("Best model:", self.best_model)

        else:
            models, scores = self._search_multiprocessing(X_train, y_train)
            self.best_model, best_score = self._find_best_model(models, scores)

            if self.verbosity:
                print("Best score:", *best_score)
                print("Best model:", self.best_model)

        self.best_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def export(self, filename):
        if self.best_model:
            pickle.dump(self.best_model, open(filename, "wb"))
