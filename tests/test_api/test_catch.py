import copy
import pytest
import math
import numpy as np
import pandas as pd

from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.experiment import BaseExperiment, add_catch
from hyperactive.search_config import SearchConfig


search_config = SearchConfig(
    x0=list(np.arange(-10, 10, 1)),
)


def test_catch_1():
    class Experiment(BaseExperiment):
        @add_catch({TypeError: np.nan})
        def objective_function(self, access):
            a = 1 + "str"

            return 0

    experiment = Experiment()

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()


def test_catch_2():
    class Experiment(BaseExperiment):
        @add_catch({ValueError: np.nan})
        def objective_function(self, access):
            math.sqrt(-10)

            return 0

    experiment = Experiment()

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()


def test_catch_3():
    class Experiment(BaseExperiment):
        @add_catch({ZeroDivisionError: np.nan})
        def objective_function(self, access):
            x = 1 / 0

            return 0

    experiment = Experiment()

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()


def test_catch_all_0():
    class Experiment(BaseExperiment):
        @add_catch(
            {
                TypeError: np.nan,
                ValueError: np.nan,
                ZeroDivisionError: np.nan,
            }
        )
        def objective_function(self, access):
            a = 1 + "str"
            math.sqrt(-10)
            x = 1 / 0

            return 0

    experiment = Experiment()

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()

    nan_ = hyper.search_data(experiment)["score"].values[0]

    assert math.isnan(nan_)


def test_catch_all_1():
    catch_return = (np.nan, {"error": True})

    class Experiment(BaseExperiment):
        @add_catch(
            {
                TypeError: catch_return,
                ValueError: catch_return,
                ZeroDivisionError: catch_return,
            }
        )
        def objective_function(self, access):
            a = 1 + "str"
            math.sqrt(-10)
            x = 1 / 0

            return 0, {"error": False}

    experiment = Experiment()

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100,
    )
    hyper.run()

    nan_ = hyper.search_data(experiment)["score"].values[0]
    error_ = hyper.search_data(experiment)["error"].values[0]

    assert math.isnan(nan_)
    assert error_ == True
