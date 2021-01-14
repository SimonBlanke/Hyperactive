import os
import inspect
import pytest

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd

from hyperactive import Hyperactive, LongTermMemory

data = load_iris()
X, y = data.data, data.target


def func1():
    pass


def func2():
    pass


def func3():
    pass


class class1:
    pass


class class2:
    pass


class class3:
    pass


class class1_:
    def __init__(self):
        pass


class class2_:
    def __init__(self):
        pass


class class3_:
    def __init__(self):
        pass


search_space_int0 = {
    "x1": list(range(2, 30, 1)),
}

search_space_int1 = {
    "x1": list(range(2, 30, 1)),
    "x2": list(range(0, 101, 1)),
}

search_space_int2 = {
    "x1": list(range(2, 30, 1)),
    "x2": list(range(-100, 1, 1)),
}

search_space_float = {
    "x1": list(range(2, 30, 1)),
    "x2": list(np.arange(0, 0.003, 0.001)),
}

search_space_str = {
    "x1": list(range(2, 30, 1)),
    "str1": ["0", "1", "2"],
}

search_space_func = {
    "x1": list(range(2, 30, 1)),
    "func1": [func1, func2, func3],
}


search_space_class = {
    "x1": list(range(2, 30, 1)),
    "class1": [class1, class2, class3],
}


search_space_obj = {
    "x1": list(range(2, 30, 1)),
    "class1": [class1_(), class2_(), class3_()],
}

search_space_lists = {
    "x1": list(range(2, 30, 1)),
    "list1": [[1, 1, 1], [1, 2, 1], [1, 1, 2]],
}


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


def model(para):
    knr = KNeighborsClassifier(n_neighbors=para["x1"])
    scores = cross_val_score(knr, X, y, cv=2)
    score = scores.mean()

    return score


def keras_model(para):
    pass


search_space_para = (
    "search_space",
    [
        (search_space_int0),
        (search_space_int1),
        (search_space_int2),
        (search_space_float),
        (search_space_str),
        (search_space_func),
        (search_space_class),
        (search_space_obj),
        (search_space_lists),
    ],
)

path_para = (
    "path",
    [("."), ("./"), (None), ("./dir/dir/")],
)


objective_function_para = (
    "objective_function",
    [(objective_function), (model),],
)


@pytest.mark.parametrize(*objective_function_para)
@pytest.mark.parametrize(*path_para)
@pytest.mark.parametrize(*search_space_para)
def test_ltm_0(objective_function, search_space, path):
    model_name = str(objective_function.__name__)
    """
    array = np.arange(3 * 10).reshape(10, 3)
    array = array / 1000
    results1 = pd.DataFrame(array, columns=["x1", "x2", "x3"])
    """

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=10, initialize={"random": 1}
    )
    hyper.run()
    results1 = hyper.results(objective_function)

    memory = LongTermMemory(model_name, path=path)
    memory.save(results1, objective_function)
    results2 = memory.load()

    print("\n results1 \n", results1)
    print("\n results2 \n", results2)

    memory.remove_model_data()

    assert results1.equals(results2)
