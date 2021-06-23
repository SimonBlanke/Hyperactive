import pytest
import numbers
import numpy as np
import pandas as pd

from hyperactive import Hyperactive


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


def objective_function1(opt):
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100, 1),
}


def test_attributes_best_score_objective_function_0():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(objective_function), numbers.Number)


def test_attributes_best_score_objective_function_1():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.add_search(
        objective_function1,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(objective_function), numbers.Number)


def test_attributes_best_score_search_id_0():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        search_id="1",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(objective_function), numbers.Number)


def test_attributes_best_score_search_id_1():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        search_id="1",
        n_iter=15,
    )
    hyper.add_search(
        objective_function1,
        search_space,
        search_id="2",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(objective_function), numbers.Number)


def test_attributes_best_para_objective_function_0():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para(objective_function), dict)


def test_attributes_best_para_objective_function_1():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.add_search(
        objective_function1,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para(objective_function), dict)


def test_attributes_best_para_search_id_0():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        search_id="1",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para("1"), dict)


def test_attributes_best_para_search_id_1():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        search_id="1",
        n_iter=15,
    )
    hyper.add_search(
        objective_function1,
        search_space,
        search_id="2",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para("1"), dict)


def test_attributes_results_objective_function_0():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.results(objective_function), pd.DataFrame)


def test_attributes_results_objective_function_1():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.add_search(
        objective_function1,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.results(objective_function), pd.DataFrame)


def test_attributes_results_search_id_0():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        search_id="1",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.results("1"), pd.DataFrame)


def test_attributes_results_search_id_1():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        search_id="1",
        n_iter=15,
    )
    hyper.add_search(
        objective_function1,
        search_space,
        search_id="2",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.results("1"), pd.DataFrame)


def test_attributes_result_errors_0():
    with pytest.raises(ValueError):
        hyper = Hyperactive()
        hyper.add_search(objective_function, search_space, n_iter=15)
        hyper.run()

        hyper.best_para(objective_function1)


def test_attributes_result_errors_1():
    with pytest.raises(ValueError):
        hyper = Hyperactive()
        hyper.add_search(objective_function, search_space, n_iter=15)
        hyper.run()

        hyper.best_score(objective_function1)


def test_attributes_result_errors_2():
    with pytest.raises(ValueError):
        hyper = Hyperactive()
        hyper.add_search(objective_function, search_space, n_iter=15)
        hyper.run()

        hyper.results(objective_function1)
