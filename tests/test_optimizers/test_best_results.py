import pytest
import numpy as np


from hyperactive import Hyperactive
from ._parametrize import optimizers


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_m5(opt):
    score = -(opt["x1"] - 5) * (opt["x1"] - 5)
    return score


def objective_function_p5(opt):
    score = -(opt["x1"] + 5) * (opt["x1"] + 5)
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


objective_para = (
    "objective",
    [
        (objective_function, search_space),
        (objective_function_m5, search_space),
        (objective_function_p5, search_space),
    ],
)


@pytest.mark.parametrize(*objective_para)
@pytest.mark.parametrize(*optimizers)
def test_best_results_0(Optimizer, objective):
    search_space = objective[1]
    objective_function = objective[0]

    initialize = {"vertices": 2}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=Optimizer(),
        n_iter=30,
        memory=False,
        initialize=initialize,
    )
    hyper.run()

    assert hyper.best_score(objective_function) == objective_function(
        hyper.best_para(objective_function)
    )


@pytest.mark.parametrize(*optimizers)
def test_best_results_1(Optimizer):
    search_space = {"x1": np.arange(-100, 101, 1)}

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    initialize = {"vertices": 2}

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=Optimizer(),
        n_iter=30,
        memory=False,
        initialize=initialize,
    )
    hyper.run()

    assert hyper.best_para(objective_function)["x1"] in list(
        hyper.results(objective_function)["x1"]
    )

