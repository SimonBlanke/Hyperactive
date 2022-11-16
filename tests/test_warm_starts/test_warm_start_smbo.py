import time
import pytest
import numpy as np
import pandas as pd

from hyperactive import (
    Hyperactive,
)

from hyperactive.optimizers import (
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)


def func1():
    pass


def func2():
    pass


class class1:
    def __init__(self):
        pass


class class2:
    def __init__(self):
        pass


def class_f1():
    return class1


def class_f2():
    return class2


def numpy_f1():
    return np.array([0, 1])


def numpy_f2():
    return np.array([1, 0])


search_space = {
    "x0": list(range(-3, 3)),
    "x1": list(np.arange(-1, 1, 0.001)),
    "string0": ["str0", "str1"],
    "function0": [func1, func2],
    "class0": [class_f1, class_f2],
    "numpy0": [numpy_f1, numpy_f2],
}


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


smbo_opts = [
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
]


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_0(smbo_opt):
    hyper0 = Hyperactive()
    hyper0.add_search(objective_function, search_space, n_iter=15)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive()
    hyper1.add_search(objective_function, search_space, n_iter=15, optimizer=smbo_opt_)
    hyper1.run()


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_1(smbo_opt):
    hyper0 = Hyperactive(distribution="pathos")
    hyper0.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive()
    hyper1.add_search(objective_function, search_space, n_iter=15, optimizer=smbo_opt_)
    hyper1.run()


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_2(smbo_opt):
    hyper0 = Hyperactive()
    hyper0.add_search(objective_function, search_space, n_iter=15)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive(distribution="joblib")
    hyper1.add_search(
        objective_function, search_space, n_iter=15, n_jobs=2, optimizer=smbo_opt_
    )
    hyper1.run()


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_3(smbo_opt):
    hyper0 = Hyperactive(distribution="pathos")
    hyper0.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive(distribution="joblib")
    hyper1.add_search(
        objective_function, search_space, n_iter=15, n_jobs=2, optimizer=smbo_opt_
    )
    hyper1.run()
