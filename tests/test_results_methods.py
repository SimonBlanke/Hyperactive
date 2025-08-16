"""Test module for results methods functionality."""

import numbers

import numpy as np
import pandas as pd
import pytest

from hyperactive import Hyperactive


def objective_function(opt):
    """Primary objective function for results testing."""
    score = -opt["x1"] * opt["x1"]
    return score


def objective_function1(opt):
    """Secondary objective function for results testing."""
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": list(np.arange(0, 100, 1)),
}


def test_attributes_best_score_objective_function_0():
    """Test best score returns numeric value."""
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_score(objective_function), numbers.Number)


def test_attributes_best_score_objective_function_1():
    """Test best score with multiple objective functions."""
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


"""
def test_attributes_best_score_search_id_0():
    # Test best score with search ID.
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
    # Test best score with multiple search IDs.
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
"""


def test_attributes_best_para_objective_function_0():
    """Test best parameters returns dictionary."""
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.best_para(objective_function), dict)


def test_attributes_best_para_objective_function_1():
    """Test best parameters with multiple objective functions."""
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


"""
def test_attributes_best_para_search_id_0():
    # Test best parameters with search ID.
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
    # Test best parameters with multiple search IDs.
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
"""


def test_attributes_results_objective_function_0():
    """Test search results returns DataFrame."""
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)


def test_attributes_results_objective_function_1():
    """Test search results with multiple objective functions."""
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

    assert isinstance(hyper.search_data(objective_function), pd.DataFrame)


"""
def test_attributes_results_search_id_0():
    # Test search results with search ID.
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        search_id="1",
        n_iter=15,
    )
    hyper.run()

    assert isinstance(hyper.search_data("1"), pd.DataFrame)


def test_attributes_results_search_id_1():
    # Test search results with multiple search IDs.
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

    assert isinstance(hyper.search_data("1"), pd.DataFrame)
"""


def test_attributes_result_errors_0():
    """Test error handling with no search runs."""
    with pytest.raises(ValueError):
        hyper = Hyperactive()
        hyper.add_search(objective_function, search_space, n_iter=15)
        hyper.run()

        hyper.best_para(objective_function1)


def test_attributes_result_errors_1():
    """Test error handling with unknown objective function."""
    with pytest.raises(ValueError):
        hyper = Hyperactive()
        hyper.add_search(objective_function, search_space, n_iter=15)
        hyper.run()

        hyper.best_score(objective_function1)


def test_attributes_result_errors_2():
    """Test error handling with unknown search ID."""
    with pytest.raises(ValueError):
        hyper = Hyperactive()
        hyper.add_search(objective_function, search_space, n_iter=15)
        hyper.run()

        hyper.search_data(objective_function1)
