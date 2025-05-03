import pytest
import numpy as np


from hyperactive import Hyperactive
from hyperactive.optimizers.strategies import CustomOptimizationStrategy

from ._parametrize import optimizers, optimizers_strat


def objective_function(opt):
    score = -(opt["x1"] * opt["x1"] + opt["x2"] * opt["x2"])
    return score


search_space = {
    "x1": list(np.arange(-3, 3, 1)),
    "x2": list(np.arange(-3, 3, 1)),
}


@pytest.mark.parametrize(*optimizers)
@pytest.mark.parametrize(*optimizers_strat)
def test_strategy_combinations_0(Optimizer, Optimizer_strat):
    optimizer1 = Optimizer()
    optimizer2 = Optimizer_strat()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.5)
    opt_strat.add_optimizer(optimizer2, duration=0.5)

    n_iter = 30

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=False,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 15
    assert len(optimizer2.search_data) == 15

    assert optimizer1.best_score <= optimizer2.best_score


@pytest.mark.parametrize(*optimizers)
@pytest.mark.parametrize(*optimizers_strat)
def test_strategy_combinations_1(Optimizer, Optimizer_strat):
    optimizer1 = Optimizer()
    optimizer2 = Optimizer_strat()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.1)
    opt_strat.add_optimizer(optimizer2, duration=0.9)

    n_iter = 10

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=False,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 1
    assert len(optimizer2.search_data) == 9

    assert optimizer1.best_score <= optimizer2.best_score


@pytest.mark.parametrize(*optimizers)
@pytest.mark.parametrize(*optimizers_strat)
def test_strategy_combinations_2(Optimizer, Optimizer_strat):
    optimizer1 = Optimizer()
    optimizer2 = Optimizer_strat()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.9)
    opt_strat.add_optimizer(optimizer2, duration=0.1)

    n_iter = 10

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=False,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 9
    assert len(optimizer2.search_data) == 1

    assert optimizer1.best_score <= optimizer2.best_score


@pytest.mark.parametrize(*optimizers)
@pytest.mark.parametrize(*optimizers_strat)
def test_strategy_combinations_3(Optimizer, Optimizer_strat):
    optimizer1 = Optimizer()
    optimizer2 = Optimizer_strat()
    optimizer3 = Optimizer_strat()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=10)
    opt_strat.add_optimizer(optimizer2, duration=20)
    opt_strat.add_optimizer(optimizer3, duration=30)

    n_iter = 100

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=False,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]
    optimizer3 = hyper.opt_pros[0].optimizer_setup_l[2]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == round(n_iter * 10 / 60)
    assert len(optimizer2.search_data) == round(n_iter * 20 / 60)
    assert len(optimizer3.search_data) == round(n_iter * 30 / 60)

    assert optimizer1.best_score <= optimizer2.best_score <= optimizer3.best_score
