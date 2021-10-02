import numpy as np
from hyperactive import Hyperactive

""" --- test search spaces with mixed int/float types --- """


def test_mixed_type_search_space_0():
    def objective_function(para):
        assert isinstance(para["x1"], int)

        return 1

    search_space = {
        "x1": list(range(10, 20)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=10000)
    hyper.run()


def test_mixed_type_search_space_1():
    def objective_function(para):
        assert isinstance(para["x2"], float)

        return 1

    search_space = {
        "x2": list(np.arange(1, 2, 0.1)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=10000)
    hyper.run()


def test_mixed_type_search_space_2():
    def objective_function(para):
        assert isinstance(para["x1"], int)
        assert isinstance(para["x2"], float)

        return 1

    search_space = {
        "x1": list(range(10, 20)),
        "x2": list(np.arange(1, 2, 0.1)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=10000)
    hyper.run()


def test_mixed_type_search_space_3():
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
    hyper.add_search(objective_function, search_space, n_iter=10000)
    hyper.run()
