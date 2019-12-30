# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

data = load_iris()
X_np = data.data
y_np = data.target

iris_X_train_columns = ["x1", "x2", "x3", "x4"]
X_pd = pd.DataFrame(X_np, columns=iris_X_train_columns)
y_pd = pd.DataFrame(y_np, columns=["y1"])


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=3)

    return scores.mean()


search_config = {
    model: {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}

"""
def test_data():
    from hyperactive import Hyperactive

    opt0 = Hyperactive(search_config)
    opt0.search(X_np, y_np)

    opt1 = Hyperactive(search_config)
    opt1.search(X_pd, y_pd)
"""
