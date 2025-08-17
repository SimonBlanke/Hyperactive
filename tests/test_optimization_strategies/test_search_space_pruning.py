"""Test module for search space pruning optimization strategy."""

import time

import numpy as np
import pytest

from hyperactive import Hyperactive
from hyperactive.optimizers import GridSearchOptimizer
from hyperactive.optimizers.strategies import CustomOptimizationStrategy

from ._parametrize import optimizers_smbo


@pytest.mark.parametrize(*optimizers_smbo)
def test_memory_Warm_start_smbo_0(Optimizer_smbo):
    """Test memory warm start with SMBO optimizers and custom optimization strategy."""

    def objective_function(opt):
        score = -(opt["x1"] * opt["x1"])
        return score

    search_space = {
        "x1": list(np.arange(0, 20, 1)),
    }

    optimizer1 = GridSearchOptimizer()
    optimizer2 = Optimizer_smbo()

    opt_strat = CustomOptimizationStrategy()

    duration_1 = 0.8
    duration_2 = 0.2

    opt_strat.add_optimizer(optimizer1, duration=duration_1)
    opt_strat.add_optimizer(optimizer2, duration=duration_2)

    n_iter = 10

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

    assert len(optimizer1.search_data) == int(n_iter * duration_1)
    assert len(optimizer2.search_data) == int(n_iter * duration_2)

    assert optimizer1.best_score <= optimizer2.best_score
