# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from hyperactive import RandomSearchOptimizer

import pytest

def test_import_hyperactive():
    dim_X = 1
    dim_Y = 1
    num_train = 5
    num_test = 10
    
    X = np.zeros((num_train, dim_X))
    Y = np.ones((num_train, dim_Y))
    X_test = np.zeros((num_test, dim_X))
    
    search_config = {
        "sklearn.ensemble.RandomForestClassifier": {"n_estimators": range(10, 100, 10)}
    }

    Optimizer = RandomSearchOptimizer(search_config, n_iter=10, verbosity=0)
    Optimizer.fit(X, y)
    
    self.assertIsNotNone(Optimizer.predict(X_test))
