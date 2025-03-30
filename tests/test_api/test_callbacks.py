import copy
import pytest
import numpy as np
import pandas as pd

from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.experiment import BaseExperiment, add_callback
from hyperactive.search_config import SearchConfig


search_config = SearchConfig(
    x0=list(np.arange(-10, 10, 1)),
)


def test_callback_0():
    class Experiment(BaseExperiment):
        def callback_1(self, access):
            access.stuff1 = 1

        def callback_2(self, access):
            access.stuff2 = 2

        @add_callback(before=[callback_1, callback_2])
        def objective_function(self, access):
            assert access.stuff1 == 1
            assert access.stuff2 == 2

            return 0

    experiment = Experiment()

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=20,
    )
    hyper.run()


def test_callback_1():
    class Experiment(BaseExperiment):
        def callback_1(self, access):
            access.stuff1 = 1

        def callback_2(self, access):
            access.stuff1 = 2

        @add_callback(before=[callback_1], after=[callback_2])
        def objective_function(self, access):
            assert access.stuff1 == 1

            return 0

    experiment = Experiment()

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()


def test_callback_2():
    class Experiment(BaseExperiment):

        def callback_1(self, access):
            self.test_var = 1

        def setup(self, test_var):
            self.test_var = test_var

        @add_callback(before=[callback_1])
        def objective_function(self, access):
            assert self.test_var == 1

            return 0

    experiment = Experiment()
    experiment.setup(5)

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()


def test_callback_3():
    class Experiment(BaseExperiment):

        def callback_1(self, access):
            access.pass_through["stuff1"] = 1

        def setup(self, test_var):
            self.test_var = test_var

        @add_callback(after=[callback_1])
        def objective_function(self, access):
            if access.nth_iter == 0:
                assert self.test_var == 0
            else:
                assert self.test_var == 1

            return 0

    experiment = Experiment()
    experiment.setup(0)

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()
