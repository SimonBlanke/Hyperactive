# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target


def model(para, X, y):
    model = DecisionTreeClassifier(
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}

warm_start = {model: {"max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 2}}


def test_func_return():
    def model1(para, X, y):
        model = DecisionTreeClassifier(
            criterion=para["criterion"],
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            min_samples_leaf=para["min_samples_leaf"],
        )
        scores = cross_val_score(model, X, y, cv=3)

        return scores.mean(), model

    search_config1 = {
        model1: {
            "criterion": ["gini", "entropy"],
            "max_depth": range(1, 21),
            "min_samples_split": range(2, 21),
            "min_samples_leaf": range(1, 21),
        }
    }

    opt = Hyperactive(search_config1)
    opt.search(X, y)


def test_n_jobs_2():
    opt = Hyperactive(search_config, n_jobs=2)
    opt.search(X, y)


def test_n_jobs_4():
    opt = Hyperactive(search_config, n_jobs=4)
    opt.search(X, y)


def test_positional_args():
    opt0 = Hyperactive(search_config, random_state=False)
    opt0.search(X, y)

    opt1 = Hyperactive(search_config, random_state=1)
    opt1.search(X, y)

    opt2 = Hyperactive(search_config, random_state=1)
    opt2.search(X, y)


def test_random_state():
    opt0 = Hyperactive(search_config, random_state=False)
    opt0.search(X, y)

    opt1 = Hyperactive(search_config, random_state=0)
    opt1.search(X, y)

    opt2 = Hyperactive(search_config, random_state=1)
    opt2.search(X, y)


def test_max_time():
    opt0 = Hyperactive(search_config, max_time=0.001)
    opt0.search(X, y)


def test_memory():
    opt0 = Hyperactive(search_config, memory=True)
    opt0.search(X, y)

    opt1 = Hyperactive(search_config, memory=False)
    opt1.search(X, y)


def test_verbosity():
    opt0 = Hyperactive(search_config, verbosity=0)
    opt0.search(X, y)

    opt0 = Hyperactive(search_config, n_jobs=2, verbosity=0)
    opt0.search(X, y)

    opt1 = Hyperactive(search_config, verbosity=1)
    opt1.search(X, y)

    opt0 = Hyperactive(search_config, n_jobs=2, verbosity=1)
    opt0.search(X, y)

    opt1 = Hyperactive(search_config, verbosity=2)
    opt1.search(X, y)


def test_scatter_init():
    opt = Hyperactive(search_config, scatter_init=10)
    opt.search(X, y)


def test_optimizer_args():
    opt = Hyperactive(search_config, optimizer={"HillClimbing": {"epsilon": 0.1}})
    opt.search(X, y)


def test_scatter_init_and_warm_start():
    opt = Hyperactive(search_config, warm_start=warm_start, scatter_init=10)
    opt.search(X, y)

    opt = Hyperactive(search_config, warm_start=warm_start, scatter_init=10)
    opt.search(X, y)


def test_warm_start_multiple_jobs():
    opt = Hyperactive(search_config, n_jobs=4, warm_start=warm_start)
    opt.search(X, y)


def test_warm_start():
    opt = Hyperactive(search_config, n_jobs=1, warm_start=warm_start)
    opt.search(X, y)


def test_get_search_path():
    opt = Hyperactive(search_config, get_search_path=True)
    opt.search(X, y)

    opt = Hyperactive(search_config, optimizer="ParticleSwarm", get_search_path=True)
    opt.search(X, y)


def test_load_memory():
    para = pd.DataFrame(
        np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]),
        columns=[
            "N_columns",
            "N_rows",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
        ],
    )
    score = pd.DataFrame(np.array([1, 1]), columns=["mean_test_score"])

    opt = Hyperactive(search_config, n_iter=3, memory="long")
    opt.search(X, y)
    opt._optimizer_.search(0, X, y)._space_.load_memory(para, score)
    
