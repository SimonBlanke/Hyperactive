import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive, HillClimbingOptimizer


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100000, 0.1),
}


def test_max_time_0():
    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    c_time1 = time.time()
    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=1000000, memory=False,
    )
    hyper.run(max_time=0.1)
    diff_time1 = time.time() - c_time1

    assert diff_time1 < 1


def test_max_time_1():
    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    c_time1 = time.time()
    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=1000000, memory=False,
    )
    hyper.run(max_time=1)
    diff_time1 = time.time() - c_time1

    assert 0.3 < diff_time1 < 2


def test_max_score_0():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 100, 0.1),
    }

    max_score = -9999

    opt = HillClimbingOptimizer(epsilon=0.01, rand_rest_p=0)

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

    print("\n Results head \n", hyper.results(objective_function).head())
    print("\n Results tail \n", hyper.results(objective_function).tail())

    print("\nN iter:", len(hyper.results(objective_function)))

    assert -100 > hyper.best_score(objective_function) > max_score


def test_max_score_1():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        time.sleep(0.01)
        return score

    search_space = {
        "x1": np.arange(0, 100, 0.1),
    }

    max_score = -9999

    c_time = time.time()
    opt = HillClimbingOptimizer()

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

    diff_time = time.time() - c_time

    print("\n Results head \n", hyper.results(objective_function).head())
    print("\n Results tail \n", hyper.results(objective_function).tail())

    print("\nN iter:", len(hyper.results(objective_function)))

    assert diff_time < 1

