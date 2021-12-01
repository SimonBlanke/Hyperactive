import numpy as np
from hyperactive import Hyperactive


def objective_function(opt):
    score = -(opt["x1"] * opt["x1"] + opt["x2"] * opt["x2"])
    return score


search_space = {
    "x1": list(np.arange(-1000, 1000, 0.1)),
    "x2": list(np.arange(-1000, 1000, 0.1)),
}


err = 0.001


def test_random_state_n_jobs_0():
    n_jobs = 2

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=5,
        initialize={"random": 1},
        random_state=1,
        n_jobs=n_jobs,
    )
    hyper.run()

    results = hyper.search_data(objective_function)

    no_dup = results.drop_duplicates(subset=list(search_space.keys()))
    print("no_dup", no_dup)
    print("results", results)

    print(int(len(results) / n_jobs))
    print(len(no_dup))

    assert int(len(results) / n_jobs) != len(no_dup)


def test_random_state_n_jobs_1():
    n_jobs = 3

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=5,
        initialize={"random": 1},
        random_state=1,
        n_jobs=n_jobs,
    )
    hyper.run()

    results = hyper.search_data(objective_function)

    no_dup = results.drop_duplicates(subset=list(search_space.keys()))
    print("no_dup", no_dup)
    print("results", results)

    assert int(len(results) / n_jobs) != len(no_dup)


def test_random_state_n_jobs_2():
    n_jobs = 4

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=5,
        initialize={"random": 1},
        random_state=1,
        n_jobs=n_jobs,
    )
    hyper.run()

    results = hyper.search_data(objective_function)

    no_dup = results.drop_duplicates(subset=list(search_space.keys()))
    print("no_dup", no_dup)
    print("results", results)

    assert int(len(results) / n_jobs) != len(no_dup)


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
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
    )
    hyper0.run()

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=10,
        initialize={"random": 1},
    )
    hyper1.run()

    best_score0 = hyper0.best_score(objective_function)
    best_score1 = hyper1.best_score(objective_function)

    assert abs(best_score0 - best_score1) > err
