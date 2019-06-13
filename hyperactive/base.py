# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import pickle
import multiprocessing
import random

import scipy
import numpy as np


from sklearn.metrics import accuracy_score
from functools import partial

from .model import MachineLearner
from .model import DeepLearner


class BaseOptimizer(object):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
    ):

        self.search_config = search_config
        self.n_iter = n_iter
        self.metric = metric
        self.memory = memory
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbosity = verbosity
        self.random_state = random_state
        self.start_points = start_points

        self.X_train = None
        self.y_train = None
        self.model_type = None

        self._get_model_type()

        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        self._set_n_jobs()

        self._n_process_range = range(0, self.n_jobs)

    def _set_random_seed(self, thread=0):
        if self.random_state:
            random.seed(self.random_state + thread)
            np.random.seed(self.random_state + thread)
            scipy.random.seed(self.random_state + thread)

    def _is_all_same(self, list):
        if len(set(list)) == 1:
            return True
        else:
            return False

    def _get_model_type(self):
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

    def _get_sklearn_model(self, n_process):
        if self.n_models > self.n_jobs:
            diff = self.n_models - self.n_jobs

            if n_process == 0:
                print(
                    "\nNot enough jobs to process models. The last",
                    diff,
                    "models will not be processed",
                )
            model_key = self.model_list[n_process]
        elif n_process < self.n_models:
            model_key = self.model_list[n_process]
        else:
            model_key = random.choice(self.model_list)

        return model_key

    def _set_n_jobs(self):
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores
        if self.n_jobs > self.n_iter:
            self.n_iter = self.n_jobs

    def _set_n_steps(self, n_process):
        n_steps = int(self.n_iter / self.n_jobs)
        remain = self.n_iter % self.n_jobs

        if n_process < remain:
            n_steps += 1

        return n_steps

    def _find_best_model(self, models, scores):
        N_best_models = 1

        scores = np.array(scores)
        index_best_scores = scores.argsort()[-N_best_models:][::-1]

        best_score = scores[index_best_scores]
        best_model = models[index_best_scores[0]]

        return best_model, best_score

    def _init_search(self, n_process, X_train, y_train):
        if self.model_type == "sklearn" or self.model_type == "xgboost":

            model_module_str = self._get_sklearn_model(n_process)
            _, self.model_str = model_module_str.rsplit(".", 1)

            self.search_space_inst.create_mlSearchSpace(
                self.search_config, model_module_str
            )

            self.model = MachineLearner(
                self.search_config, self.metric, self.cv, model_module_str
            )
            self.metric_type = self.model._get_metric_type_sklearn()

            hyperpara_indices = self.search_space_inst.init_eval(n_process, "sklearn")
        elif self.model_type == "keras":
            self.model_str = "keras model"
            self.search_space_inst.create_kerasSearchSpace(self.search_config)
            self.model = DeepLearner(self.search_config, self.metric, self.cv)

            self.metric_type = self.model._get_metric_type_keras()

            hyperpara_indices = self.search_space_inst.init_eval(n_process, "keras")

        return hyperpara_indices

    def _finish_search(self, best_hyperpara_dict, n_process):
        if self.model_type == "sklearn" or self.model_type == "xgboost":
            start_point = self.model.create_start_point(best_hyperpara_dict, n_process)
        elif self.model_type == "keras":
            start_point = self.model.trafo_hyperpara_dict_lists(best_hyperpara_dict)

        return start_point

    def _search_normalprocessing(self, X_train, y_train):
        best_model, best_score, start_point = self._search(0, X_train, y_train)

        return best_model, best_score, start_point

    def _search_multiprocessing(self, X_train, y_train):
        pool = multiprocessing.Pool(self.n_jobs)

        _search = partial(self._search, X_train=X_train, y_train=y_train)

        best_models, scores, start_points = zip(
            *pool.map(_search, self._n_process_range)
        )

        self.best_model_list = best_models
        self.score_list = scores

        return best_models, scores, start_points

    def fit(self, X_train, y_train):
        if self.model_type == "keras":
            self.n_jobs = 1

        if self.n_jobs == 1:
            self.best_model, best_score, start_point = self._search_normalprocessing(
                X_train, y_train
            )
            if self.verbosity:
                print("\nScore:", best_score)
                print("start_point =", start_point, "\n")

                # self.best_model.summery()
        else:
            models, scores, start_points = self._search_multiprocessing(
                X_train, y_train
            )

            self.best_model, best_score = self._find_best_model(models, scores)

            if self.verbosity:
                for score, start_point in zip(scores, start_points):
                    print("\n", self.metric, best_score)
                    print("start_point =", start_point, "\n")

        self.best_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def score(self, X_test, y_test):
        if self.model_type == "sklearn":
            y_pred = self.predict(X_test)
            return accuracy_score(y_pred, y_test)
        elif self.model_type == "keras":
            return self.best_model.evaluate(X_test, y_test)[1]

    def export(self, filename):
        if self.best_model:
            pickle.dump(self.best_model, open(filename, "wb"))
