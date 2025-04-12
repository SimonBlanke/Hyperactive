import pytest
import numbers
import numpy as np
import pandas as pd

from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.experiment import BaseExperiment
from hyperactive.search_config import SearchConfig


class Experiment(BaseExperiment):
    def objective_function(self, opt):
        score = -opt["x1"] * opt["x1"]
        return score


class Experiment1(BaseExperiment):
    def objective_function(self, opt):
        score = -opt["x1"] * opt["x1"]
        return score


experiment = Experiment()
experiment1 = Experiment1()


search_config = SearchConfig(
    x1=list(np.arange(0, 100, 1)),
)


def test_attributes_best_score_objective_function_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(experiment), numbers.Number)


def test_attributes_best_score_objective_function_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.add_search(
        experiment1,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(experiment), numbers.Number)


"""
def test_attributes_best_score_search_id_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        search_id="1",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(experiment), numbers.Number)


def test_attributes_best_score_search_id_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        search_id="1",
        n_iter=15,
    )
    hyper.add_search(
        experiment1,
        search_config,
        search_id="2",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(experiment), numbers.Number)
"""


def test_attributes_best_para_objective_function_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para(experiment), dict)


def test_attributes_best_para_objective_function_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.add_search(
        experiment1,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para(experiment), dict)


"""
def test_attributes_best_para_search_id_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        search_id="1",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para("1"), dict)


def test_attributes_best_para_search_id_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        search_id="1",
        n_iter=15,
    )
    hyper.add_search(
        experiment1,
        search_config,
        search_id="2",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para("1"), dict)
"""


def test_attributes_results_objective_function_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)


def test_attributes_results_objective_function_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=15,
    )
    hyper.add_search(
        experiment1,
        search_config,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(experiment), pd.DataFrame)


"""
def test_attributes_results_search_id_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        search_id="1",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data("1"), pd.DataFrame)


def test_attributes_results_search_id_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        search_id="1",
        n_iter=15,
    )
    hyper.add_search(
        experiment1,
        search_config,
        search_id="2",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data("1"), pd.DataFrame)
"""


def test_attributes_result_errors_0():
    with pytest.raises(ValueError):
        hyper = HillClimbingOptimizer()
        hyper.add_search(experiment, search_config, n_iter=15)
        hyper.run()

        hyper.best_para(experiment1)


def test_attributes_result_errors_1():
    with pytest.raises(ValueError):
        hyper = HillClimbingOptimizer()
        hyper.add_search(experiment, search_config, n_iter=15)
        hyper.run()

        hyper.best_score(experiment1)


def test_attributes_result_errors_2():
    with pytest.raises(ValueError):
        hyper = HillClimbingOptimizer()
        hyper.add_search(experiment, search_config, n_iter=15)
        hyper.run()

        hyper.search_data(experiment1)
