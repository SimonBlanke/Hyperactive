# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from hyperactive import Hyperactive

data = load_iris()
X, y = data.data, data.target


def model(para, X, y):
    dtc = DecisionTreeClassifier(
        max_depth=para["max_depth"], min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(dtc, X, y, cv=2)

    return scores.mean()


search_space = {
    "max_depth": range(1, 21),
    "min_samples_split": range(2, 21),
}


def _base_test(search, opt_args={}, time=None):
    opt = Hyperactive(X, y, **opt_args)
    opt.add_search(**search)
    opt.run(time)


def test_HillClimbingOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "HillClimbing",
    }

    _base_test(search)


def test_StochasticHillClimbingOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "StochasticHillClimbing",
    }

    _base_test(search)


def test_TabuOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "TabuSearch",
    }

    _base_test(search)


def test_RandomSearchOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "RandomSearch",
    }

    _base_test(search)


def test_RandomRestartHillClimbingOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "RandomRestartHillClimbing",
    }

    _base_test(search)


def test_RandomAnnealingOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "RandomAnnealing",
    }

    _base_test(search)


def test_SimulatedAnnealingOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "SimulatedAnnealing",
    }

    _base_test(search)


def test_StochasticTunnelingOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "StochasticTunneling",
    }

    _base_test(search)


def test_ParallelTemperingOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "ParallelTempering",
    }

    _base_test(search)


def test_ParticleSwarmOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "ParticleSwarm",
    }

    _base_test(search)


def test_EvolutionStrategyOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "EvolutionStrategy",
    }

    _base_test(search)


def test_BayesianOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "Bayesian",
    }

    _base_test(search)


def test_TPE():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "TPE",
    }

    _base_test(search)


def test_DecisionTreeOptimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
        "optimizer": "DecisionTree",
    }

    _base_test(search)
