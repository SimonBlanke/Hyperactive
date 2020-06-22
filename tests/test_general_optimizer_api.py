# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from hyperactive import Optimizer

data = load_iris()
X, y = data.data, data.target


def objective_function(para):
    dtc = DecisionTreeClassifier(
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(dtc, para["features"], para["target"], cv=2)

    return scores.mean()


search_space = {
    "max_depth": range(1, 21),
    "min_samples_split": range(2, 21),
    "min_samples_leaf": range(1, 21),
}

search = {
    "objective_function": objective_function,
    "function_parameter": {"features": X, "target": y},
    "search_space": search_space,
}


def _base_test(search):
    opt = Optimizer()
    opt.add_search(**search)
    opt.run()


def test_n_jobs():
    n_jobs_list = [1, 2, 4, 10, 100, -1]
    for n_jobs in n_jobs_list:
        search["n_jobs"] = n_jobs
        _base_test(search)


def test_positional_args():
    opt = Optimizer()
    opt.add_search(objective_function, search_space)
    opt.run()


def test_n_iter():
    n_iter_list = [0, 1, 2, 4, 10, 100]
    for n_iter in n_iter_list:
        search["n_iter"] = n_iter
        _base_test(search)


test_n_iter()
"""
def test_optimizer():
    optimizer_list = [0, 1, 2, 4, 10, 100]
    for optimizer in optimizer_list:
        search["optimizer"] = optimizer
        _base_test(search)
"""
