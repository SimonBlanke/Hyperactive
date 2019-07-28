# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from sklearn.model_selection import cross_val_score

from .metrics import ml_scores, ml_losses
from .model import Model


class MachineLearner(Model):
    def __init__(self, search_config, metric, cv, model_str):
        super().__init__(search_config, metric, cv)

        self.search_config = search_config
        self.metric = metric
        self.cv = cv
        self.model_str = model_str

        self.model = self._get_model(model_str)

        self.scores = ml_scores
        self.losses = ml_losses

        self._get_metric_type_sklearn()

    def _get_metric_type_sklearn(self):
        if self.metric in self.scores:
            self.metric_type = "score"
        elif self.metric in self.losses:
            self.metric_type = "loss"

    def create_start_point(self, sklearn_para_dict, nth_process):
        start_point = {}
        model_str = self.model_str + "." + str(nth_process)

        temp_dict = {}
        for para_key in sklearn_para_dict:
            temp_dict[para_key] = [sklearn_para_dict[para_key]]

        start_point[model_str] = temp_dict

        return start_point

    def _create_model(self, sklearn_para_dict):
        return self.model(**sklearn_para_dict), 1

    def train_model(self, sklearn_para_dict, X_train, y_train):
        sklearn_model, _ = self._create_model(sklearn_para_dict)

        time_temp = time.perf_counter()
        scores = cross_val_score(
            sklearn_model, X_train, y_train, scoring=self.metric, cv=self.cv
        )
        train_time = (time.perf_counter() - time_temp) / self.cv

        if self.metric_type == "score":
            return scores.mean(), train_time, sklearn_model
        elif self.metric_type == "loss":
            return -scores.mean(), train_time, sklearn_model
