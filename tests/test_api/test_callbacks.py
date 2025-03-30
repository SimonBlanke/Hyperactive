import copy
import pytest
import numpy as np
import pandas as pd

from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.base import BaseExperiment
from hyperactive.search_config import SearchConfig


def test_callback_0():
    def callback_1(access):
        access.stuff1 = 1

    def callback_2(access):
        access.stuff2 = 2

    class Experiment(BaseExperiment):
        def objective_function(self, access):
            assert access.stuff1 == 1
            assert access.stuff2 == 2

            return 0

    search_config = SearchConfig(
        x0=list(np.arange(2, 15, 1)),
        x1=list(np.arange(2, 25, 2)),
    )

    objective_function = Experiment(callbacks={"before": [callback_1, callback_2]})

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        objective_function,
        search_config,
        n_iter=20,
    )
    hyper.run()
