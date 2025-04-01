import numpy as np
import pandas as pd

from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.experiment import BaseExperiment
from hyperactive.search_config import SearchConfig


class Experiment(BaseExperiment):
    def objective_function(self, opt):
        score = -opt["x1"] * opt["x1"]
        return score


experiment = Experiment()


def test_search_space_0():
    search_config = SearchConfig(
        x1=list(np.arange(0, 3, 1)),
    )

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)
    assert hyper.best_para(experiment)["x1"] in search_config["x1"]


def test_search_space_1():
    search_config = SearchConfig(
        x1=list(np.arange(0, 0.003, 0.001)),
    )

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)
    assert hyper.best_para(experiment)["x1"] in search_config["x1"]


def test_search_space_2():
    search_config = SearchConfig(
        x1=list(np.arange(0, 100, 1)),
        str1=["0", "1", "2"],
    )

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)
    assert hyper.best_para(experiment)["str1"] in search_config["str1"]


def test_search_space_3():
    def func1():
        pass

    def func2():
        pass

    def func3():
        pass

    search_config = SearchConfig(
        x1=list(np.arange(0, 100, 1)),
        func1=[func1, func2, func3],
    )

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)
    assert hyper.best_para(experiment)["func1"] in search_config["func1"]


def test_search_space_4():
    class class1:
        pass

    class class2:
        pass

    class class3:
        pass

    search_config = SearchConfig(
        x1=list(np.arange(0, 100, 1)),
        class1=[class1, class2, class3],
    )

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)
    assert hyper.best_para(experiment)["class1"] in search_config["class1"]


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

    search_config = SearchConfig(
        x1=list(np.arange(0, 100, 1)),
        class1=[class_f1, class_f2, class_f3],
    )

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)
    assert hyper.best_para(experiment)["class1"] in search_config["class1"]


def test_search_space_6():

    def list_f1():
        return [0, 1]

    def list_f2():
        return [1, 0]

    search_config = SearchConfig(
        x1=list(np.arange(0, 100, 1)),
        list1=[list_f1, list_f2],
    )

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)
    assert hyper.best_para(experiment)["list1"] in search_config["list1"]
