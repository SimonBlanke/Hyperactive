# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

search_config = {
    "xgboost.XGBClassifier": {
        "n_estimators": range(30, 200, 10),
        "max_depth": range(1, 11),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "subsample": np.arange(0.05, 1.01, 0.05),
        "min_child_weight": range(1, 21),
        "nthread": [1],
    }
}


def test_xgboost():
    from hyperactive import SimulatedAnnealingOptimizer

    opt0 = SimulatedAnnealingOptimizer(search_config, n_iter=20, verbosity=0)
    opt0.fit(X, y)


test_xgboost()
