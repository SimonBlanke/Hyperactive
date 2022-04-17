import copy
import pytest
import numpy as np
import pandas as pd

from hyperactive import Hyperactive


search_space = {
    "x1": list(np.arange(-100, 100, 1)),
}


def callback_1(access):
    access.stuff1 = 1


def callback_2(access):
    access.stuff2 = 2


def objective_function(access):
    access.stuff1 == 1
    access.stuff2 == 2

    return 0


def test_callback_0():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        callbacks={"before": [callback_1, callback_2]},
    )
    hyper.run()


def callback_1(access):
    access.stuff1 = 1


def callback_2(access):
    access.stuff1 = 2


def objective_function(access):
    access.stuff1 == 1

    return 0


def test_callback_1():
    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        callbacks={"before": [callback_1], "after": [callback_2]},
    )
    hyper.run()
