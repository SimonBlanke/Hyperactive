import numpy as np
from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.experiment import BaseExperiment
from hyperactive.search_config import SearchConfig


class Experiment(BaseExperiment):
    def objective_function(self, opt):
        score = -opt["x1"] * opt["x1"]
        return score


experiment = Experiment()


search_config = SearchConfig(
    x1=list(np.arange(-100, 101, 1)),
)


def test_initialize_warm_start_0():
    init = {
        "x1": 0,
    }

    initialize = {"warm_start": [init]}

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=1,
        initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(experiment)) < 0.001


def test_initialize_warm_start_1():
    search_space = {
        "x1": list(np.arange(-10, 10, 1)),
    }
    init = {
        "x1": -10,
    }

    initialize = {"warm_start": [init]}

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=1,
        initialize=initialize,
    )
    hyper.run()

    assert hyper.best_para(experiment) == init


def test_initialize_vertices():
    initialize = {"vertices": 2}

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=2,
        initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(experiment)) - 10000 < 0.001


def test_initialize_grid_0():
    search_space = {
        "x1": list(np.arange(-1, 2, 1)),
    }
    initialize = {"grid": 1}

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=1,
        initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(experiment)) < 0.001


def test_initialize_grid_1():
    search_space = {
        "x1": list(np.arange(-2, 3, 1)),
    }

    initialize = {"grid": 1}

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=1,
        initialize=initialize,
    )
    hyper.run()

    assert abs(hyper.best_score(experiment)) - 1 < 0.001


def test_initialize_all_0():
    search_space = {
        "x1": list(np.arange(-2, 3, 1)),
    }

    initialize = {"grid": 100, "vertices": 100, "random": 100}

    hyper = HillClimbingOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=300,
        initialize=initialize,
    )
    hyper.run()
