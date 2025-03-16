import copy
import pytest
import numpy as np
import pandas as pd

from hyperactive.optimizers import HillClimbingOptimizer


from .test_setup import SphereFunction, search_config


objective_function = SphereFunction()


def test_callback_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(
        objective_function,
        search_config,
        n_iter=100,
    )
    hyper.run()
