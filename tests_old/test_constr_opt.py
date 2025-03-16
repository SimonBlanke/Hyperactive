import numpy as np

from hyperactive import Hyperactive


def test_constr_opt_0():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": list(np.arange(-15, 15, 1)),
    }

    def constraint_1(para):
        print(" para", para)

        return para["x1"] > -5

    constraints_list = [constraint_1]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=50,
        constraints=constraints_list,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)
    x0_values = search_data["x1"].values

    print("\n search_data \n", search_data, "\n")

    assert np.all(x0_values > -5)


def test_constr_opt_1():
    def objective_function(para):
        score = -(para["x1"] * para["x1"] + para["x2"] * para["x2"])
        return score

    search_space = {
        "x1": list(np.arange(-10, 10, 1)),
        "x2": list(np.arange(-10, 10, 1)),
    }

    def constraint_1(para):
        return para["x1"] > -5

    constraints_list = [constraint_1]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=50,
        constraints=constraints_list,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)
    x0_values = search_data["x1"].values

    print("\n search_data \n", search_data, "\n")

    assert np.all(x0_values > -5)


def test_constr_opt_2():
    n_iter = 50

    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": list(np.arange(-10, 10, 0.1)),
    }

    def constraint_1(para):
        return para["x1"] > -5

    def constraint_2(para):
        return para["x1"] < 5

    constraints_list = [constraint_1, constraint_2]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=50,
        constraints=constraints_list,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)
    x0_values = search_data["x1"].values

    print("\n search_data \n", search_data, "\n")

    assert np.all(x0_values > -5)
    assert np.all(x0_values < 5)

    n_new_positions = 0
    n_new_scores = 0

    n_current_positions = 0
    n_current_scores = 0

    n_best_positions = 0
    n_best_scores = 0

    for hyper_optimizer in hyper.opt_pros.values():
        optimizer = hyper_optimizer.gfo_optimizer

        n_new_positions = n_new_positions + len(optimizer.pos_new_list)
        n_new_scores = n_new_scores + len(optimizer.score_new_list)

        n_current_positions = n_current_positions + len(optimizer.pos_current_list)
        n_current_scores = n_current_scores + len(optimizer.score_current_list)

        n_best_positions = n_best_positions + len(optimizer.pos_best_list)
        n_best_scores = n_best_scores + len(optimizer.score_best_list)

        print("\n  optimizer", optimizer)
        print("  n_new_positions", optimizer.pos_new_list)
        print("  n_new_scores", optimizer.score_new_list)

    assert n_new_positions == n_iter
    assert n_new_scores == n_iter

    assert n_current_positions == n_current_scores
    assert n_current_positions <= n_new_positions

    assert n_best_positions == n_best_scores
    assert n_best_positions <= n_new_positions

    assert n_new_positions == n_new_scores
