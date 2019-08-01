# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from .model_sklearn import ScikitLearnModel


class LightGbmModel(ScikitLearnModel):
    def __init__(self, _config_, search_config_key):
        super().__init__(_config_, search_config_key)

    def _create_model(self, para_dict):
        return self.model(**para_dict)

    def _cross_val_lightgbm(self, model, X, y):
        scores = []

        kf = KFold(n_splits=self.cv, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = self.metric_class(y_test, y_pred)
            scores.append(score)

        return np.array(scores).mean(), model

    def train_model(self, para, X, y):
        lightgbm_model = self._create_model(para)

        if self.cv > 1:
            score, model = self._cross_val_lightgbm(lightgbm_model, X, y)
        elif self.cv < 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=self.cv
            )

            model = lightgbm_model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = self.metric_class(y_test, y_pred)
        else:
            score = 0
            model = lightgbm_model.fit(X, y)

        if self.metric_type == "score":
            return score, model
        elif self.metric_type == "loss":
            return -score, model
