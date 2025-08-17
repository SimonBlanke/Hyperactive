# ruff: noqa: D100, D103
import sys

import numpy as np
import pytest

from hyperactive import (
    Hyperactive,
)
from hyperactive.optimizers import (
    BayesianOptimizer,
    ForestOptimizer,
    TreeStructuredParzenEstimators,
)

if sys.platform.startswith("win"):
    pytest.skip("skip these tests for windows", allow_module_level=True)


def _func1():
    pass


def _func2():
    pass


class _class1:
    def __init__(self):
        pass


class _class2:
    def __init__(self):
        pass


def _class_f1():
    return _class1


def _class_f2():
    return _class2


def _numpy_f1():
    return np.array([0, 1])


def _numpy_f2():
    return np.array([1, 0])


search_space = {
    "x0": list(range(-3, 3)),
    "x1": list(np.arange(-1, 1, 0.001)),
    "string0": ["str0", "str1"],
    "function0": [_func1, _func2],
    "class0": [_class_f1, _class_f2],
    "numpy0": [_numpy_f1, _numpy_f2],
}


def _objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


smbo_opts = [
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
]

initialize = {"random": 1}
n_iter = 3


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_0(smbo_opt):
    hyper0 = Hyperactive()
    hyper0.add_search(_objective_function, search_space, n_iter=n_iter)
    hyper0.run()

    search_data0 = hyper0.search_data(_objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive()
    hyper1.add_search(
        _objective_function,
        search_space,
        n_iter=n_iter,
        optimizer=smbo_opt_,
        initialize=initialize,
    )
    hyper1.run()


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_1(smbo_opt):
    hyper0 = Hyperactive(distribution="pathos")
    hyper0.add_search(
        _objective_function,
        search_space,
        n_iter=n_iter,
        n_jobs=2,
        initialize=initialize,
    )
    hyper0.run()

    search_data0 = hyper0.search_data(_objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive()
    hyper1.add_search(
        _objective_function, search_space, n_iter=n_iter, optimizer=smbo_opt_
    )
    hyper1.run()


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_2(smbo_opt):
    hyper0 = Hyperactive()
    hyper0.add_search(_objective_function, search_space, n_iter=n_iter)
    hyper0.run()

    search_data0 = hyper0.search_data(_objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive(distribution="joblib")
    hyper1.add_search(
        _objective_function,
        search_space,
        n_iter=n_iter,
        n_jobs=2,
        optimizer=smbo_opt_,
        initialize=initialize,
    )
    hyper1.run()


@pytest.mark.parametrize("smbo_opt", smbo_opts)
def test_warm_start_smbo_3(smbo_opt):
    hyper0 = Hyperactive(distribution="pathos")
    hyper0.add_search(_objective_function, search_space, n_iter=n_iter, n_jobs=2)
    hyper0.run()

    search_data0 = hyper0.search_data(_objective_function)
    smbo_opt_ = smbo_opt(warm_start_smbo=search_data0)

    hyper1 = Hyperactive(distribution="joblib")
    hyper1.add_search(
        _objective_function,
        search_space,
        n_iter=n_iter,
        n_jobs=2,
        optimizer=smbo_opt_,
        initialize=initialize,
    )
    hyper1.run()
