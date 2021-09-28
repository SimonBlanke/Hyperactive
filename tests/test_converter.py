#!/usr/bin/env python3

import pytest
import numpy as np

from hyperactive import Hyperactive


def test_mixed_type_search_space():
    """Test search spaces with mixed int/float types. Issue #34."""

    def objective_function(para):
        nonlocal para_types
        for v, t in zip(para.values(), para_types):
            assert isinstance(v, t)
        score = 0
        for x1 in range(para["x1"]):
            score += -(x1 ** 2) + para["x2"] + 100
        return score

    search_space = {
        "x1": list(range(10, 20)),
        "x2": list(np.arange(1, 2, 0.1)),
    }
    para_types = [int, float]
    expected_pos = [1, 9]

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    best_para = hyper.best_para(objective_function)
    for best_para_val, expected_p, dim_space, p_type in zip(
        best_para.values(), expected_pos, search_space.values(), para_types
    ):
        assert best_para_val == dim_space[expected_p]
        assert isinstance(best_para_val, p_type)
