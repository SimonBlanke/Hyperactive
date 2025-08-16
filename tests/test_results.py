"""Test module for results functionality."""

import numpy as np
import pandas as pd
import pytest

from hyperactive import Hyperactive


def objective_function(opt):
    """Return simple quadratic objective function for results testing."""
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": list(np.arange(0, 100, 1)),
}


def test_attributes_results_0():
    """Test search data returns pandas DataFrame."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)


def test_attributes_results_1():
    """Test search data contains search space columns."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert set(search_space.keys()) < set(hyper.search_data(objective_function).columns)


def test_attributes_results_2():
    """Test search data contains x1 column."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert "x1" in list(hyper.search_data(objective_function).columns)


def test_attributes_results_3():
    """Test search data contains score column."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    assert "score" in list(hyper.search_data(objective_function).columns)


def test_attributes_results_4():
    """Test warm start initialization with specific value."""
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=1,
        initialize={"warm_start": [{"x1": 0}]},
    )
    hyper.run()

    assert 0 in list(hyper.search_data(objective_function)["x1"].values)


def test_attributes_results_5():
    """Test warm start initialization with different value."""
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
        list(hyper.search_data(objective_function)["x1"].values),
    )

    assert 10 in list(hyper.search_data(objective_function)["x1"].values)


def test_attributes_results_6():
    """Test memory disabled allows duplicate search space points."""
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
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

    x1_results = list(hyper.search_data(objective_function)["x1"].values)

    print("\n x1_results \n", x1_results)

    assert len(set(x1_results)) < len(x1_results)


def test_attributes_results_7():
    """Test search data without times parameter excludes timing columns."""
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)
    with pytest.raises(Exception):
        search_data["eval_times"]


def test_attributes_results_8():
    """Test search data without times parameter excludes iteration timing."""
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)
    with pytest.raises(Exception):
        search_data["iter_times"]


def test_attributes_results_9():
    """Test search data with times parameter includes timing columns."""
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function, times=True)
    search_data["iter_times"]
    search_data["eval_times"]


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
