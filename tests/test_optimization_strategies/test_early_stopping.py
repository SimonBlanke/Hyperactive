import pytest
import numpy as np


from hyperactive import Hyperactive
from hyperactive.optimizers.strategies import CustomOptimizationStrategy
from hyperactive.optimizers import RandomSearchOptimizer

from ._parametrize import optimizers


n_iter_no_change_parametr = (
    "n_iter_no_change",
    [
        (5),
        (10),
        (15),
    ],
)


@pytest.mark.parametrize(*n_iter_no_change_parametr)
@pytest.mark.parametrize(*optimizers)
def test_strategy_early_stopping_0(Optimizer, n_iter_no_change):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": list(np.arange(0, 100, 0.1)),
    }

    # n_iter_no_change = 5
    early_stopping = {
        "n_iter_no_change": n_iter_no_change,
    }

    optimizer1 = Optimizer()
    optimizer2 = RandomSearchOptimizer()

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(optimizer1, duration=0.9, early_stopping=early_stopping)
    opt_strat.add_optimizer(optimizer2, duration=0.1)

    n_iter = 30

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=opt_strat,
        n_iter=n_iter,
        initialize={"warm_start": [{"x1": 0}]},
    )
    hyper.run()

    optimizer1 = hyper.opt_pros[0].optimizer_setup_l[0]["optimizer"]
    optimizer2 = hyper.opt_pros[0].optimizer_setup_l[1]["optimizer"]

    search_data = optimizer1.search_data
    n_performed_iter = len(search_data)

    print("\n n_performed_iter \n", n_performed_iter)
    print("\n n_iter_no_change \n", n_iter_no_change)

    assert n_performed_iter == (n_iter_no_change + 1)
