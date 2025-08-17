"""Test module for exception catching functionality."""

import math

import numpy as np

from hyperactive import Hyperactive

search_space = {
    "x1": list(np.arange(-20, 20, 1)),
}


def test_catch_1():
    """Test catching TypeError exceptions in objective function."""

    def objective_function(access):
        1 + "str"  # Intentional TypeError for testing

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        catch={TypeError: np.nan},
    )
    hyper.run()


def test_catch_2():
    """Test catching ValueError exceptions in objective function."""

    def objective_function(access):
        math.sqrt(-10)

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        catch={ValueError: np.nan},
    )
    hyper.run()


def test_catch_3():
    """Test catching ZeroDivisionError exceptions in objective function."""

    def objective_function(access):
        1 / 0  # Intentional ZeroDivisionError for testing

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        catch={ZeroDivisionError: np.nan},
    )
    hyper.run()


def test_catch_all_0():
    """Test catching multiple exception types returning NaN values."""

    def objective_function(access):
        1 + "str"  # Intentional TypeError for testing
        math.sqrt(-10)  # Intentional ValueError for testing
        1 / 0  # Intentional ZeroDivisionError for testing

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        catch={
            TypeError: np.nan,
            ValueError: np.nan,
            ZeroDivisionError: np.nan,
        },
    )
    hyper.run()

    nan_ = hyper.search_data(objective_function)["score"].values[0]

    assert math.isnan(nan_)


def test_catch_all_1():
    """Test catching multiple exception types returning tuple values."""

    def objective_function(access):
        1 + "str"  # Intentional TypeError for testing
        math.sqrt(-10)  # Intentional ValueError for testing
        1 / 0  # Intentional ZeroDivisionError for testing

        return 0, {"error": False}

    catch_return = (np.nan, {"error": True})

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        catch={
            TypeError: catch_return,
            ValueError: catch_return,
            ZeroDivisionError: catch_return,
        },
    )
    hyper.run()

    nan_ = hyper.search_data(objective_function)["score"].values[0]
    error_ = hyper.search_data(objective_function)["error"].values[0]

    assert math.isnan(nan_)
    assert error_
