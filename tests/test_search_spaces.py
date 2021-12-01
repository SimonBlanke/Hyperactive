import numpy as np
import pandas as pd

from hyperactive import Hyperactive


def test_search_space_0():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(range(0, 3, 1)),
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)
    assert hyper.best_para(objective_function)["x1"] in search_space["x1"]


def test_search_space_1():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 0.003, 0.001)),
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)
    assert hyper.best_para(objective_function)["x1"] in search_space["x1"]


def test_search_space_2():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(range(0, 100, 1)),
        "str1": ["0", "1", "2"],
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)
    assert hyper.best_para(objective_function)["str1"] in search_space["str1"]


def test_search_space_3():
    def func1():
        pass

    def func2():
        pass

    def func3():
        pass

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(range(0, 100, 1)),
        "func1": [func1, func2, func3],
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)
    assert hyper.best_para(objective_function)["func1"] in search_space["func1"]


def test_search_space_4():
    class class1:
        pass

    class class2:
        pass

    class class3:
        pass

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(range(0, 100, 1)),
        "class1": [class1, class2, class3],
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)
    assert hyper.best_para(objective_function)["class1"] in search_space["class1"]


def test_search_space_5():
    class class1:
        def __init__(self):
            pass

    class class2:
        def __init__(self):
            pass

    class class3:
        def __init__(self):
            pass

    def class_f1():
        return class1

    def class_f2():
        return class2

    def class_f3():
        return class3

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(range(0, 100, 1)),
        "class1": [class_f1, class_f2, class_f3],
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)
    assert hyper.best_para(objective_function)["class1"] in search_space["class1"]


def test_search_space_6():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    def list_f1():
        return [0, 1]

    def list_f2():
        return [1, 0]

    search_space = {
        "x1": list(range(0, 100, 1)),
        "list1": [list_f1, list_f2],
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)
    assert hyper.best_para(objective_function)["list1"] in search_space["list1"]
