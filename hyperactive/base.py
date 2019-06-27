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

from .model import DeepLearner
from .positioner import MlCandidate
from .positioner import DlCandidate


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
        warm_start=False,
    ):

        self.search_config = search_config
        self.n_iter = n_iter
        self.metric = metric
        self.memory = memory
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbosity = verbosity
        self.random_state = random_state
        self.warm_start = warm_start

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

    def _get_sklearn_model(self, nth_process):
        if self.n_models > self.n_jobs:
            diff = self.n_models - self.n_jobs

            if nth_process == 0:
                print(
                    "\nNot enough jobs to process models. The last",
                    diff,
                    "models will not be processed",
                )
            model_key = self.model_list[nth_process]
        elif nth_process < self.n_models:
            model_key = self.model_list[nth_process]
        else:
            model_key = random.choice(self.model_list)

        return model_key

    def _set_n_jobs(self):
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores
        if self.n_jobs > self.n_iter:
            self.n_iter = self.n_jobs

    def _set_n_steps(self, nth_process):
        n_steps = int(self.n_iter / self.n_jobs)
        remain = self.n_iter % self.n_jobs

        if nth_process < remain:
            n_steps += 1

        return n_steps

    def _find_best_model(self, models, scores, n_best_models=1):
        scores = np.array(scores)
        index_best_scores = list(scores.argsort()[-n_best_models:][::-1])

        best_score = scores[index_best_scores]
        best_model = models[index_best_scores]

        if n_best_models == 1:
            best_score = best_score[0]
            best_model = best_model[0]

        return best_model, best_score

    def _init_search(self, nth_process, X, y):
        self._set_random_seed(nth_process)
        self.n_steps = self._set_n_steps(nth_process)

        if self.model_type == "sklearn" or self.model_type == "xgboost":

            search_config_key = self._get_sklearn_model(nth_process)
            __cand_ = MlCandidate(
                nth_process,
                self.search_config,
                False,
                self.metric,
                self.cv,
                search_config_key,
            )

        elif self.model_type == "keras":
            self._dl_cand_ = DlCandidate(nth_process, self.search_config, False)

            if self.warm_start:
                return self.space.warm_start_dl(nth_process)
            else:
                return self.space.get_random_position()

        return __cand_

    def _init_population_search(self, nth_process, X, y, n_candidates):
        hyperpara_indices_list = []

        if self.model_type == "sklearn" or self.model_type == "xgboost":
            self.space.set_warm_start()
            self._init_ml_search(nth_process, X, y)

            for candidate in n_candidates:
                if candidate == 0:
                    hyperpara_indices = self.space.warm_start_ml(nth_process)
                else:
                    hyperpara_indices = self.space.get_random_position()

                hyperpara_indices_list.append(hyperpara_indices)

        elif self.model_type == "keras":
            self._init_dl_search(nth_process, X, y)

            for candidate in n_candidates:
                if candidate == 0:
                    hyperpara_indices = self.space.warm_start_dl(nth_process)
                else:
                    hyperpara_indices = self.space.get_random_position()

                hyperpara_indices_list.append(hyperpara_indices)

        return hyperpara_indices_list

    def _init_dl_search(self, nth_process, X_train, y_train):
        self.model_str = "keras model"
        self.space.create_kerasSearchSpace(self.search_config)
        self.model = DeepLearner(self.search_config, self.metric, self.cv)

        self.metric_type = self.model._get_metric_type_keras()

    def _finish_search(self, best_hyperpara_dict, nth_process):
        if self.model_type == "sklearn" or self.model_type == "xgboost":
            start_point = self.model.create_start_point(
                best_hyperpara_dict, nth_process
            )
        elif self.model_type == "keras":
            start_point = self.model.trafo_hyperpara_dict_lists(best_hyperpara_dict)

        return start_point

    def _search_normalprocessing(self, X_train, y_train):
        best_model, best_score, start_point = self.search(0, X_train, y_train)

        return best_model, best_score, start_point

    def _search_multiprocessing(self, X_train, y_train):
        pool = multiprocessing.Pool(self.n_jobs)

        search = partial(self.search, X=X_train, y=y_train)

        best_models, scores, warm_start = zip(*pool.map(search, self._n_process_range))

        self.best_model_list = best_models
        self.score_list = scores

        return best_models, scores, warm_start

    def fit(self, X_train, y_train):
        if self.model_type == "keras":
            self.n_jobs = 1

        if self.n_jobs == 1:
            self.best_model, self.best_score, start_point = self._search_normalprocessing(
                X_train, y_train
            )
            if self.verbosity:
                print("\n", self.metric, self.best_score)
                print("start_point =", start_point)

                # self.best_model.summery()
        else:
            models, scores, warm_start = self._search_multiprocessing(X_train, y_train)

            warm_start = list(warm_start)
            warm_start = np.array(warm_start)

            models = list(models)
            models = np.array(models)

            warm_starts, score_best = self._find_best_model(
                warm_start, scores, n_best_models=self.n_jobs
            )

            self.pos_best, self.score_best = self._find_best_model(models, scores)

            print("\nList of start points (best first):")
            if self.verbosity:
                for score, warm_start in zip(score_best, warm_starts):
                    print("\n", self.metric, score)
                    print("warm_start =", warm_start)

        # self.best_model.fit(X_train, y_train)

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


class BaseCandidate:
    def __init__(self, model):
        self.model = model
        self.hyperpara_dict = None
        self.score = 0

    def set_position(self, hyperpara_dict):
        self.hyperpara_dict = hyperpara_dict

    def eval(self, X_train, y_train):
        self.score, _, self.sklearn_model = self.model.train_model(
            self.hyperpara_dict, X_train, y_train
        )
