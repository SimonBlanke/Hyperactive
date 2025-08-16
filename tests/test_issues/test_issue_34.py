"""Test module for issue #34 reproduction."""

import numpy as np

from hyperactive import Hyperactive

""" --- test search spaces with mixed int/float types --- """
n_iter = 100


def test_mixed_type_search_space_0():
    """Test search space with integer type validation."""
    def objective_function(para):
        assert isinstance(para["x1"], int)

        return 1

    search_space = {
        "x1": list(range(10, 20)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=n_iter)
    hyper.run()


def test_mixed_type_search_space_1():
    """Test search space with float type validation."""
    def objective_function(para):
        assert isinstance(para["x2"], float)

        return 1

    search_space = {
        "x2": list(np.arange(1, 2, 0.1)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=n_iter)
    hyper.run()


def test_mixed_type_search_space_2():
    """Test search space with mixed integer and float type validation."""
    def objective_function(para):
        assert isinstance(para["x1"], int)
        assert isinstance(para["x2"], float)

        return 1

    search_space = {
        "x1": list(range(10, 20)),
        "x2": list(np.arange(1, 2, 0.1)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=n_iter)
    hyper.run()


def test_mixed_type_search_space_3():
    """Test search space with mixed integer, float, and string type validation."""
    def objective_function(para):
        assert isinstance(para["x1"], int)
        assert isinstance(para["x2"], float)
        assert isinstance(para["x3"], float)
        assert isinstance(para["x4"], str)

        return 1

    search_space = {
        "x1": list(range(10, 20)),
        "x2": list(np.arange(1, 2, 0.1)),
        "x3": list(np.arange(1, 2, 0.1)),
        "x4": ["str1", "str2", "str3"],
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=n_iter)
    hyper.run()
