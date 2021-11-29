import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from hyperactive import Hyperactive
from hyperactive.optimizers import (
    RandomSearchOptimizer,
    HillClimbingOptimizer,
)


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": list(np.arange(0, 100000, 0.1)),
}


def test_max_score_0():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 100, 0.1)),
    }

    max_score = -9999

    opt = HillClimbingOptimizer(
        epsilon=0.01,
        rand_rest_p=0,
    )

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt,
        n_iter=100000,
        initialize={"warm_start": [{"x1": 99}]},
        max_score=max_score,
    )
    hyper.run()

    print("\n Results head \n", hyper.search_data(objective_function).head())
    print("\n Results tail \n", hyper.search_data(objective_function).tail())

    print("\nN iter:", len(hyper.search_data(objective_function)))

    assert -100 > hyper.best_score(objective_function) > max_score


def test_max_score_1():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        time.sleep(0.01)
        return score

    search_space = {
        "x1": list(np.arange(0, 100, 0.1)),
    }

    max_score = -9999

    c_time = time.perf_counter()
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100000,
        initialize={"warm_start": [{"x1": 99}]},
        max_score=max_score,
    )
    hyper.run()
    diff_time = time.perf_counter() - c_time

    print("\n Results head \n", hyper.search_data(objective_function).head())
    print("\n Results tail \n", hyper.search_data(objective_function).tail())

    print("\nN iter:", len(hyper.search_data(objective_function)))

    assert diff_time < 1
