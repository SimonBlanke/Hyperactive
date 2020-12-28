import numpy as np
from hyperactive import Hyperactive


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": np.arange(0, 1000, 1),
}


err = 0.001


def test_random_state_0():
    hyper0 = Hyperactive()
    hyper0.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        random_state=1,
    )
    hyper0.run()

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        random_state=1,
    )
    hyper1.run()

    best_score0 = hyper0.best_score(objective_function)
    best_score1 = hyper1.best_score(objective_function)

    assert abs(best_score0 - best_score1) < err


def test_random_state_1():
    hyper0 = Hyperactive()
    hyper0.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        random_state=10,
    )
    hyper0.run()

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        random_state=10,
    )
    hyper1.run()

    best_score0 = hyper0.best_score(objective_function)
    best_score1 = hyper1.best_score(objective_function)

    assert abs(best_score0 - best_score1) < err


def test_random_state_2():
    hyper0 = Hyperactive()
    hyper0.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        random_state=1,
    )
    hyper0.run()

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
        random_state=10,
    )
    hyper1.run()

    best_score0 = hyper0.best_score(objective_function)
    best_score1 = hyper1.best_score(objective_function)

    assert abs(best_score0 - best_score1) > err


def test_no_random_state_0():
    hyper0 = Hyperactive()
    hyper0.add_search(
        objective_function, search_space, n_iter=10, initialize={"random": 1},
    )
    hyper0.run()

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function, search_space, n_iter=10, initialize={"random": 1},
    )
    hyper1.run()

    best_score0 = hyper0.best_score(objective_function)
    best_score1 = hyper1.best_score(objective_function)

    assert abs(best_score0 - best_score1) > err
