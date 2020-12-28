import numpy as np
import pandas as pd

from hyperactive import Hyperactive


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100, 1),
}


def test_attributes_results_0():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert isinstance(hyper.results(objective_function), pd.DataFrame)


def test_attributes_results_1():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert set(search_space.keys()) < set(
        hyper.results(objective_function).columns
    )


def test_attributes_results_2():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert "x1" in list(hyper.results(objective_function).columns)


def test_attributes_results_3():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert "score" in list(hyper.results(objective_function).columns)


def test_attributes_results_4():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=1,
        initialize={"warm_start": [{"x1": 0}]},
    )
    hyper.run()

    assert 0 in list(hyper.results(objective_function)["x1"].values)


def test_attributes_results_5():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=1,
        initialize={"warm_start": [{"x1": 10}]},
    )
    hyper.run()

    print(
        "\n x1_results \n",
        list(hyper.results(objective_function)["x1"].values),
    )

    assert 10 in list(hyper.results(objective_function)["x1"].values)


def test_attributes_results_6():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        initialize={"random": 1},
        memory=False,
    )
    hyper.run()

    x1_results = list(hyper.results(objective_function)["x1"].values)

    print("\n x1_results \n", x1_results)

    assert len(set(x1_results)) < len(x1_results)


"""
def test_attributes_results_7():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function, n_iter=20, initialize={"random": 1}, memory=True
    )

    x1_results = list(opt.results["x1"].values)

    print("\n x1_results \n", x1_results)

    assert len(set(x1_results)) == len(x1_results)


def test_attributes_results_8():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    results = pd.DataFrame(np.arange(-10, 10, 1), columns=["x1"])
    results["score"] = 0

    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function,
        n_iter=100,
        initialize={},
        memory=True,
        memory_warm_start=results,
    )

    print("\n opt.results \n", opt.results)

    x1_results = list(opt.results["x1"].values)

    assert 10 == x1_results[0]
"""
