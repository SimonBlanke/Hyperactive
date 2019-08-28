# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from importlib import import_module
from sklearn.model_selection import train_test_split

from .metrics import scores, losses


class Model:
    def __init__(self, _config_):
        self.metric = _config_.metric
        self.cv = _config_.cv

        self.scores = scores
        self.losses = losses

    def _get_model(self, model):
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model

    def train_model(self, para, X, y):
        model_ = self._create_model(para)

        if self.cv > 1:
            score, model = self._cross_val_score(model_, X, y)
        elif self.cv < 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=self.cv
            )

            model = model_.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = self.metric_class(y_test, y_pred)
        else:
            score = 0
            model = model_.fit(X, y)

        if self.metric in self.scores:
            return score, model
        elif self.metric in self.losses:
            return -score, model


class MachineLearningModel(Model):
    def __init__(self, _config_):
        super().__init__(_config_)

        # self.metric_type = list(_config_.metric.keys())[0]
        self.metric_class = self._get_metric_class(self.metric)

    def _get_metric_class(self, metric):
        module = import_module("sklearn.metrics")
        metric_class = getattr(module, metric)

        return metric_class


class DeepLearningModel(Model):
    def __init__(self, _config_):
        super().__init__(_config_)
