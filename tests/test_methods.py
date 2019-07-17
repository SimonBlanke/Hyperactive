# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from hyperactive import RandomSearchOptimizer

import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_predict():
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    search_config = {
        "sklearn.ensemble.RandomForestClassifier": {"n_estimators": range(10, 100, 10)}
    }

    Optimizer = RandomSearchOptimizer(search_config, n_iter=10, verbosity=0)
    Optimizer.fit(X_train, y_train)
    Optimizer.predict(X_test)
