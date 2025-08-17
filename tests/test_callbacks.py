"""Test module for callback functionality."""

import numpy as np

from hyperactive import Hyperactive

search_space = {
    "x1": list(np.arange(-20, 20, 1)),
}


def test_callback_0():
    """Test callbacks executed before objective function."""

    def callback_1(access):
        access.stuff1 = 1

    def callback_2(access):
        access.stuff2 = 2

    def objective_function(access):
        assert access.stuff1 == 1
        assert access.stuff2 == 2

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        callbacks={"before": [callback_1, callback_2]},
    )
    hyper.run()


def test_callback_1():
    """Test callbacks executed before and after objective function."""

    def callback_1(access):
        access.stuff1 = 1

    def callback_2(access):
        access.stuff1 = 2

    def objective_function(access):
        assert access.stuff1 == 1

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        callbacks={"before": [callback_1], "after": [callback_2]},
    )
    hyper.run()


def test_callback_2():
    """Test callbacks with pass_through parameter."""

    def callback_1(access):
        access.pass_through["stuff1"] = 1

    def objective_function(access):
        assert access.pass_through["stuff1"] == 1

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        callbacks={"before": [callback_1]},
        pass_through={"stuff1": 0},
    )
    hyper.run()


def test_callback_3():
    """Test callbacks executed after objective function with pass_through."""

    def callback_1(access):
        access.pass_through["stuff1"] = 1

    def objective_function(access):
        if access.nth_iter == 0:
            assert access.pass_through["stuff1"] == 0
        else:
            assert access.pass_through["stuff1"] == 1

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=20,
        callbacks={"after": [callback_1]},
        pass_through={"stuff1": 0},
    )
    hyper.run()
