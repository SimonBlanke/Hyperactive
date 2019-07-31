# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.model_selection import train_test_split
from importlib import import_module
from sklearn.model_selection import KFold

from .model import Model


class LightGbmModel(Model):
    def __init__(self, _config_, search_config_key):
        super().__init__(_config_)

        self.search_config_key = search_config_key
        self.model = self._get_model(search_config_key)

    def create_start_point(self, sklearn_para_dict, nth_process):
        start_point = {}
        model_str = self.search_config_key + "." + str(nth_process)

        temp_dict = {}
        for para_key in sklearn_para_dict:
            temp_dict[para_key] = [sklearn_para_dict[para_key]]

        start_point[model_str] = temp_dict

        return start_point

    def _create_model(self, para_dict):
        return self.model(**para_dict)

    def _cross_val_lightgbm(self, model, X, y, metric):
        scores = []

        kf = KFold(n_splits=self.cv, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric(y_test, y_pred)
            scores.append(score)

        return np.array(scores).mean(), model

    def train_model(self, sklearn_para_dict, X, y):
        lightgbm_model = self._create_model(sklearn_para_dict)

        module = import_module("sklearn.metrics")
        metric = getattr(module, self.metric)

        if self.cv > 1:
            score, model = self._cross_val_lightgbm(lightgbm_model, X, y, metric)
        elif self.cv < 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=self.cv
            )

            model = lightgbm_model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric(y_test, y_pred)
        else:
            score = 0
            model = lightgbm_model.fit(X, y)

        if self.metric_type == "score":
            return score, model
        elif self.metric_type == "loss":
            return -score, model
