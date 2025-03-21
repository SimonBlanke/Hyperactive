import pytest
import numpy as np


from hyperactive import Hyperactive
from hyperactive.optimizers.strategies import CustomOptimizationStrategy
from hyperactive.optimizers import HillClimbingOptimizer

from ._parametrize import optimizers


def objective_function(opt):
    score = -(opt["x1"] * opt["x1"] + opt["x2"] * opt["x2"])
    return score


search_space = {
    "x1": list(np.arange(-3, 3, 1)),
    "x2": list(np.arange(-3, 3, 1)),
}


@pytest.mark.parametrize(*optimizers)
def test_strategy_combinations_0(Optimizer):
    optimizer1 = Optimizer()
    optimizer2 = HillClimbingOptimizer()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.5)
    opt_strat.add_optimizer(optimizer2, duration=0.5)

    n_iter = 4

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=False,
        initialize={"random": 1},
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 2
    assert len(optimizer2.search_data) == 2

    assert optimizer1.best_score <= optimizer2.best_score
