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
    "x2": ["0", "1", "2"],
}

search_space_func = {
    "x1": list(range(2, 30, 1)),
    "x2": [func1, func2, func3],
}


search_space_class = {
    "x1": list(range(2, 30, 1)),
    "x2": [class1, class2, class3],
}


search_space_obj = {
    "x1": list(range(2, 30, 1)),
    "x2": [class1_(), class2_(), class3_()],
}

search_space_lists = {
    "x1": list(range(2, 30, 1)),
    "x2": [[1, 1, 1], [1, 2, 1], [1, 1, 2]],
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


def compare_0(results1, results2):
    assert results1.equals(results2)


def compare_obj(results1, results2):
    obj1_list = list(results1["x2"].values)
    obj2_list = list(results1["x2"].values)

    for obj1, obj2 in zip(obj1_list, obj2_list):
        if obj1 != obj2:
            assert False


search_space_para = (
    "search_space",
    [
        (search_space_int0, compare_0),
        (search_space_int1, compare_0),
        (search_space_int2, compare_0),
        (search_space_float, compare_0),
        (search_space_str, compare_0),
        (search_space_func, compare_obj),
        (search_space_class, compare_obj),
        (search_space_obj, compare_obj),
        (search_space_lists, compare_obj),
    ],
)

path_para = (
    "path",
    [("."), ("./"), (None), ("./dir/dir/")],
)


objective_function_para = (
    "objective_function",
    [
        (objective_function),
        (model),
    ],
)


@pytest.mark.parametrize(*objective_function_para)
@pytest.mark.parametrize(*path_para)
@pytest.mark.parametrize(*search_space_para)
def test_ltm_0(objective_function, search_space, path):
    (search_space, compare) = search_space

    print("\n objective_function \n", objective_function)
    print("\n search_space \n", search_space)
    print("\n compare \n", compare)
    print("\n path \n", path)

    model_name = str(objective_function.__name__)

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=10, initialize={"random": 1}
    )
    hyper.run()
    results1 = hyper.search_data(objective_function)

    memory = LongTermMemory(model_name, path=path)
    memory.save(results1, objective_function)
    results2 = memory.load()

    print("\n results1 \n", results1)
    print("\n results2 \n", results2)

    memory.remove_model_data()

    compare(results1, results2)


@pytest.mark.parametrize(*objective_function_para)
@pytest.mark.parametrize(*path_para)
@pytest.mark.parametrize(*search_space_para)
def test_ltm_1(objective_function, search_space, path):
    (search_space, compare) = search_space

    print("\n objective_function \n", objective_function)
    print("\n search_space \n", search_space)
    print("\n compare \n", compare)
    print("\n path \n", path)

    model_name = str(objective_function.__name__)
    memory = LongTermMemory(model_name, path=path)

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        long_term_memory=memory,
    )
    hyper1.run()
    results1 = hyper1.search_data(objective_function)

    hyper2 = Hyperactive()
    hyper2.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        long_term_memory=memory,
    )
    hyper2.run()
    results2 = hyper2.search_data(objective_function)
    memory.remove_model_data()

    print("\n results1 \n", results1)
    print("\n results2 \n", results2)
