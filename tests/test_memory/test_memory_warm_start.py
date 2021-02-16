import time
import pytest
import numpy as np
import pandas as pd

from hyperactive import Hyperactive

import numpy as np


search_space_0 = {
    "x1": list(range(0, 100, 1)),
}

search_space_1 = {
    "x1": list(range(0, 100, 1)),
    "x2": list(range(-5, 5, 1)),
    "x3": list(range(-5, 5, 1)),
}

search_space_2 = {
    "x1": list(range(0, 100, 1)),
    "x2": list(np.arange(0, 0.003, 0.001)),
}

search_space_3 = {
    "x1": list(range(0, 100, 1)),
    "str1": ["0", "1", "2"],
}


def func1():
    pass


def func2():
    pass


def func3():
    pass


search_space_3 = {
    "x1": list(range(0, 100, 1)),
    "func1": [func1, func2, func3],
}


class class1:
    pass


class class2:
    pass


class class3:
    pass


search_space_4 = {
    "x1": list(range(0, 100, 1)),
    "class1": [class1, class2, class3],
}


class class1:
    def __init__(self):
        pass


class class2:
    def __init__(self):
        pass


class class3:
    def __init__(self):
        pass


search_space_5 = {
    "x1": list(range(0, 100, 1)),
    "class1": [class1(), class2(), class3()],
}

search_space_6 = {
    "x1": list(range(0, 100, 1)),
    "list1": [[1, 1, 1], [1, 2, 1], [1, 1, 2]],
}


search_space_list = [
    (search_space_0),
    (search_space_1),
    (search_space_2),
    (search_space_3),
    (search_space_4),
    (search_space_5),
    (search_space_6),
]


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


@pytest.mark.parametrize("search_space", search_space_list)
def test_memory_warm_start_0(search_space):
    hyper0 = Hyperactive()
    hyper0.add_search(objective_function, search_space, n_iter=20)
    hyper0.run()

    search_data0 = hyper0.results(objective_function)

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function, search_space, n_iter=20, memory_warm_start=search_data0
    )
    hyper1.run()
