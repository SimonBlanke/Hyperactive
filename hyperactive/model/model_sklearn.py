# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from importlib import import_module
from sklearn.metrics import make_scorer


from .model import Model


class ScikitLearnModel(Model):
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

    def _create_model(self, sklearn_para_dict):
        return self.model(**sklearn_para_dict)

    def _train_cross_val(self, sklearn_model, X, y, metric):
        scorer = make_scorer(metric)

        scores = cross_val_score(sklearn_model, X, y, scoring=scorer, cv=self.cv)
        return scores.mean(), sklearn_model

    def train_model(self, sklearn_para_dict, X, y):
        sklearn_model = self._create_model(sklearn_para_dict)

        module = import_module("sklearn.metrics")
        metric = getattr(module, self.metric)

        if self.cv > 1:
            score, model = self._train_cross_val(sklearn_model, X, y, metric)
        elif self.cv < 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=self.cv
            )
            model = sklearn_model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric(y_test, y_pred)
        else:
            score = 0
            model = sklearn_model.fit(X, y)

        if self.metric_type == "score":
            return score, model
        elif self.metric_type == "loss":
            return -score, model
