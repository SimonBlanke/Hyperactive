# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X_np = data.data
y_np = data.target

iris_X_train_columns = ["x1", "x2", "x3", "x4"]
X_pd = pd.DataFrame(X_np, columns=iris_X_train_columns)
y_pd = pd.DataFrame(y_np, columns=["y1"])

n_iter_0 = 0
n_iter_1 = 10
random_state = 0
cv = 2

search_config = {
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}


def test_data():
    from hyperactive import HillClimbingOptimizer

    opt0 = HillClimbingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X_np, y_np)

    opt1 = HillClimbingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt1.fit(X_pd, y_pd)
