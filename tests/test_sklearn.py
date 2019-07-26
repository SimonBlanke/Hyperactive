# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

n_iter_0 = 0
n_iter_1 = 3
random_state = 0
cv = 2
n_jobs = 2

search_config = {
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(1, 80, 5),
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 11),
        "min_samples_leaf": [1],
        "subsample": np.arange(0.09, 1.01, 0.1),
        "max_features": np.arange(0.09, 1.01, 0.1),
    }
}

warm_start = {"sklearn.ensemble.GradientBoostingClassifier": {"n_estimators": [1]}}


def test_sklearn():
    from hyperactive import HillClimbingOptimizer

    opt0 = HillClimbingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)
