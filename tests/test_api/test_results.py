import pytest
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

search_config = SearchConfig(
    x1=list(np.arange(0, 100, 1)),
)


def test_attributes_results_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=100)
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)


def test_attributes_results_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=100)
    hyper.run()

    assert set(search_config.keys()) < set(hyper.search_data(experiment).columns)


def test_attributes_results_2():
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=100)
    hyper.run()

    assert "x1" in list(hyper.search_data(experiment).columns)


def test_attributes_results_3():
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=100)
    hyper.run()

    assert "score" in list(hyper.search_data(experiment).columns)


def test_attributes_results_4():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=1,
        initialize={"warm_start": [{"x1": 0}]},
    )
    hyper.run()

    assert 0 in list(hyper.search_data(experiment)["x1"].values)


def test_attributes_results_5():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=1,
        initialize={"warm_start": [{"x1": 10}]},
    )
    hyper.run()

    print(
        "\n x1_results \n",
        list(hyper.search_data(experiment)["x1"].values),
    )

    assert 10 in list(hyper.search_data(experiment)["x1"].values)


def test_attributes_results_6():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
    }

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=20,
        initialize={"random": 1},
        memory=False,
    )
    hyper.run()

    x1_results = list(hyper.search_data(experiment)["x1"].values)

    print("\n x1_results \n", x1_results)

    assert len(set(x1_results)) < len(x1_results)


def test_attributes_results_7():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
    }

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=20,
    )
    hyper.run()

    search_data = hyper.search_data(experiment)
    with pytest.raises(Exception) as e_info:
        search_data["eval_times"]


def test_attributes_results_8():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
    }

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=20,
    )
    hyper.run()

    search_data = hyper.search_data(experiment)
    with pytest.raises(Exception) as e_info:
        search_data["iter_times"]


def test_attributes_results_9():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 10, 1)),
    }

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=20,
    )
    hyper.run()

    search_data = hyper.search_data(experiment, times=True)
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
        experiment, n_iter=20, initialize={"random": 1}, memory=True
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
        experiment,
        n_iter=100,
        initialize={},
        memory=True,
        memory_warm_start=results,
    )

    print("\n opt.results \n", opt.results)

    x1_results = list(opt.results["x1"].values)

    assert 10 == x1_results[0]
"""
