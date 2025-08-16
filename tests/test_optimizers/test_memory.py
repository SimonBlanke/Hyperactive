"""Test module for optimizer memory functionality."""

import numpy as np
import pytest

from hyperactive import Hyperactive

from ._parametrize import optimizers


def objective_function(opt):
    """Return simple quadratic objective function for memory testing."""
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {"x1": list(np.arange(-10, 11, 1))}


@pytest.mark.parametrize(*optimizers)
def test_memory_0(Optimizer):
    """Test memory functionality with multiple search runs."""
    optimizer = Optimizer()

    n_iter = 30

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=optimizer,
        n_iter=n_iter,
        n_jobs=2,
    )
    hyper.add_search(
        objective_function,
        search_space,
        optimizer=optimizer,
        n_iter=n_iter,
        n_jobs=2,
    )
    hyper.run()
