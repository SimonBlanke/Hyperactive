# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target

random_state = 1
n_jobs = 2


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean(), model


search_config = {
    model: {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 11),
        "min_samples_leaf": range(1, 11),
    }
}

warm_start = {model: {"max_depth": [1]}}


def test_TabuOptimizer():
    opt0 = Hyperactive(
        search_config,
        optimizer="TabuSearch",
        n_iter=1,
        random_state=random_state,
        verbosity=0,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="TabuSearch",
        n_iter=30,
        random_state=random_state,
        verbosity=0,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best
