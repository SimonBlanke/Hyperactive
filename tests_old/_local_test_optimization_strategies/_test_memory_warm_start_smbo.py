import time
import pytest
import numpy as np


from hyperactive import Hyperactive
from hyperactive.optimizers.strategies import CustomOptimizationStrategy
from hyperactive.optimizers import GridSearchOptimizer

from ._parametrize import optimizers_smbo


def objective_function(opt):
    time.sleep(0.01)
    score = -(opt["x1"] * opt["x1"])
    return score


search_space = {
    "x1": list(np.arange(0, 100, 1)),
}


@pytest.mark.parametrize(*optimizers_smbo)
def test_memory_Warm_start_smbo_0(Optimizer_smbo):
    optimizer1 = GridSearchOptimizer()
    optimizer2 = Optimizer_smbo()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.8)
    opt_strat.add_optimizer(optimizer2, duration=0.2)

    n_iter = 100

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        memory=True,
    )
    hyper.run()

    search_data = hyper.search_data(objective_function)

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    assert len(search_data) == n_iter

    assert len(optimizer1.search_data) == 80
    assert len(optimizer2.search_data) == 20

    assert optimizer1.best_score <= optimizer2.best_score
