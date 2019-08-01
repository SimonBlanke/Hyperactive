# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.model_selection import KFold

from .model_sklearn import ScikitLearnModel


class LightGbmModel(ScikitLearnModel):
    def __init__(self, _config_, search_config_key):
        super().__init__(_config_, search_config_key)

    def _create_model(self, para_dict):
        return self.model(**para_dict)

    def _cross_val_score(self, model, X, y):
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
