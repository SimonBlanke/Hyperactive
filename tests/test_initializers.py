import numpy as np
from hyperactive import Hyperactive


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": np.arange(-100, 101, 1),
}


def test_initialize_warm_start_0():
    init = {
        "x1": 0,
    }

    initialize = {"warm_start": [init]}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=1, initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(objective_function)) < 0.001


def test_initialize_warm_start_1():
    search_space = {
        "x1": np.arange(-10, 10, 1),
    }
    init = {
        "x1": -10,
    }

    initialize = {"warm_start": [init]}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=1, initialize=initialize,
    )
    hyper.run()

    assert hyper.best_para(objective_function) == init


def test_initialize_vertices():
    initialize = {"vertices": 2}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=2, initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(objective_function)) - 10000 < 0.001


def test_initialize_grid_0():
    search_space = {
        "x1": np.arange(-1, 2, 1),
    }
    initialize = {"grid": 1}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=1, initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(objective_function)) < 0.001


def test_initialize_grid_1():
    search_space = {
        "x1": np.arange(-2, 3, 1),
    }

    initialize = {"grid": 1}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=1, initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(objective_function)) - 1 < 0.001


def test_initialize_all_0():
    search_space = {
        "x1": np.arange(-2, 3, 1),
    }

    initialize = {"grid": 100, "vertices": 100, "random": 100}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=300, initialize=initialize,
    )
    hyper.run()

