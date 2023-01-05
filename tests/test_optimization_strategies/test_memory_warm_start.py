import time
import pytest
import numpy as np


from hyperactive import Hyperactive
from hyperactive.optimizers.strategies import CustomOptimizationStrategy
from hyperactive.optimizers import GridSearchOptimizer

from ._parametrize import optimizers_non_smbo


def objective_function(opt):
    time.sleep(0.01)
    score = -(opt["x1"] * opt["x1"])
    return score


search_space = {
    "x1": list(np.arange(0, 100, 1)),
}


def test_memory_Warm_start_0():
    optimizer1 = GridSearchOptimizer()
    optimizer2 = GridSearchOptimizer()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.2)
    opt_strat.add_optimizer(optimizer2, duration=0.8)

    n_iter = 1000

    c_time = time.time()

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=True,
    )
    hyper.run()

    d_time = time.time() - c_time

    search_data = hyper.search_data(objective_function)
    
    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 200
    assert len(optimizer2.search_data) == 800

    assert optimizer1.best_score <= optimizer2.best_score

    print("\n d_time", d_time)

    assert d_time < 3


def test_memory_Warm_start_1():
    optimizer1 = GridSearchOptimizer()
    optimizer2 = GridSearchOptimizer()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.2)
    opt_strat.add_optimizer(optimizer2, duration=0.8)

    n_iter = 100

    search_space = {
        "x1": list(np.arange(0, 1, 1)),
    }

    c_time = time.time()

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=False,
    )
    hyper.run()

    d_time = time.time() - c_time

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 20
    assert len(optimizer2.search_data) == 80

    assert optimizer1.best_score <= optimizer2.best_score

    print("\n d_time", d_time)

    assert d_time > 0.95



@pytest.mark.parametrize(*optimizers_non_smbo)
def test_memory_Warm_start_2(Optimizer_non_smbo):
    optimizer1 = GridSearchOptimizer()
    optimizer2 = Optimizer_non_smbo()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.5)
    opt_strat.add_optimizer(optimizer2, duration=0.5)

    search_space = {
        "x1": list(np.arange(0, 50, 1)),
    }

    n_iter = 100

    c_time = time.time()

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=True,
    )
    hyper.run()

    d_time = time.time() - c_time

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 50
    assert len(optimizer2.search_data) == 50

    assert optimizer1.best_score <= optimizer2.best_score

    print("\n d_time", d_time)

    assert d_time < 0.9
    