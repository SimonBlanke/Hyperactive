"""Test module for memory warm start functionality."""

import sys

import numpy as np
import pytest

from hyperactive import Hyperactive

if sys.platform.startswith("win"):
    pytest.skip("skip these tests for windows", allow_module_level=True)


def func1():
    """Test function 1 for search space."""
    pass


def func2():
    """Test function 2 for search space."""
    pass


class class1:
    """Test class for search space functionality."""

    def __init__(self):
        pass


class class2:
    """Test class for search space functionality."""

    def __init__(self):
        pass


def class_f1():
    """Return class1 for search space."""
    return class1


def class_f2():
    """Return class2 for search space."""
    return class2


def numpy_f1():
    """Return numpy array [0, 1] for search space."""
    return np.array([0, 1])


def numpy_f2():
    """Return numpy array [1, 0] for search space."""
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
    """Return simple quadratic objective function for testing."""
    score = -opt["x1"] * opt["x1"]
    return score


def test_memory_warm_start_0():
    """Test memory warm start from single job to single job."""
    hyper0 = Hyperactive()
    hyper0.add_search(objective_function, search_space, n_iter=15)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=15,
        memory_warm_start=search_data0,
    )
    hyper1.run()


def test_memory_warm_start_1():
    """Test memory warm start from multi-job to single job."""
    hyper0 = Hyperactive(distribution="pathos")
    hyper0.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=15,
        memory_warm_start=search_data0,
    )
    hyper1.run()


def test_memory_warm_start_2():
    """Test memory warm start from single job to multi-job."""
    hyper0 = Hyperactive()
    hyper0.add_search(objective_function, search_space, n_iter=15)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)

    hyper1 = Hyperactive(distribution="pathos")
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=15,
        n_jobs=2,
        memory_warm_start=search_data0,
    )
    hyper1.run()


def test_memory_warm_start_3():
    """Test memory warm start from multi-job to multi-job."""
    hyper0 = Hyperactive(distribution="pathos")
    hyper0.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper0.run()

    search_data0 = hyper0.search_data(objective_function)

    hyper1 = Hyperactive(distribution="pathos")
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=15,
        n_jobs=2,
        memory_warm_start=search_data0,
    )
    hyper1.run()
