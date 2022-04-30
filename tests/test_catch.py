import copy
import pytest
import math
import numpy as np
import pandas as pd

from hyperactive import Hyperactive


search_space = {
    "x1": list(np.arange(-100, 100, 1)),
}


def test_catch_0():
    def objective_function(access):
        x = y

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={NameError: np.nan},
    )
    hyper.run()


def test_catch_1():
    def objective_function(access):
        a = 1 + "str"

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={TypeError: np.nan},
    )
    hyper.run()


def test_catch_2():
    def objective_function(access):
        math.sqrt(-10)

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={ValueError: np.nan},
    )
    hyper.run()


def test_catch_3():
    def objective_function(access):
        x = 1 / 0

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={ZeroDivisionError: np.nan},
    )
    hyper.run()


def test_catch_all_0():
    def objective_function(access):
        x = y
        a = 1 + "str"
        math.sqrt(-10)
        x = 1 / 0

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={
            NameError: np.nan,
            TypeError: np.nan,
            ValueError: np.nan,
            ZeroDivisionError: np.nan,
        },
    )
    hyper.run()

    nan_ = hyper.search_data(objective_function)["score"].values[0]

    assert math.isnan(nan_)


def test_catch_all_1():
    def objective_function(access):
        x = y
        a = 1 + "str"
        math.sqrt(-10)
        x = 1 / 0

        return 0, {"error": False}

    catch_return = (np.nan, {"error": True})

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={
            NameError: catch_return,
            TypeError: catch_return,
            ValueError: catch_return,
            ZeroDivisionError: catch_return,
        },
    )
    hyper.run()

    nan_ = hyper.search_data(objective_function)["score"].values[0]
    error_ = hyper.search_data(objective_function)["error"].values[0]

    assert math.isnan(nan_)
    assert error_ == True
