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


def test_max_time():
    search = {
        "model": model,
        "search_space": search_space,
    }
    _base_test(search, time=0.01)


def test_init_para():
    search = {
        "model": model,
        "search_space": search_space,
    }

    init_para1 = {
        "max_depth": 3,
        "min_samples_split": 3,
    }
    init_para_list = [[init_para1]]
    for init_para in init_para_list:
        search["init_para"] = init_para
        _base_test(search)


def test_verbosity():
    search = {
        "model": model,
        "search_space": search_space,
    }

    verbosity_list = [0, 1, 2, 3]
    for verbosity in verbosity_list:
        _base_test(search, opt_args={"verbosity": verbosity})


def test_n_jobs():
    search = {
        "model": model,
        "search_space": search_space,
    }

    n_jobs_list = [1, 2, 4, 10, -1]
    for n_jobs in n_jobs_list:
        search["n_jobs"] = n_jobs
        _base_test(search)


def test_positional_args():
    search = {
        "model": model,
        "search_space": search_space,
    }
    _base_test(search)


def test_n_iter():
    search = {
        "model": model,
        "search_space": search_space,
    }

    n_iter_list = [0, 1, 2, 4, 10, 100]
    for n_iter in n_iter_list:
        search["n_iter"] = n_iter
        _base_test(search)


def test_memory():
    search = {
        "model": model,
        "search_space": search_space,
    }

    memory_list = [False, "short"]
    for memory in memory_list:
        search["memory"] = memory
        _base_test(search)


def test_optimizer():
    search = {
        "model": model,
        "function_parameter": {"features": X, "target": y},
        "search_space": search_space,
        "n_iter": 15,
    }

    optimizer_list = [
        "HillClimbing",
        "StochasticHillClimbing",
        "TabuSearch",
        "RandomSearch",
        "RandomRestartHillClimbing",
        "RandomAnnealing",
        "SimulatedAnnealing",
        "StochasticTunneling",
        "ParallelTempering",
        "ParticleSwarm",
        "EvolutionStrategy",
        "Bayesian",
        "TPE",
        "DecisionTree",
    ]
    for optimizer in optimizer_list:
        search["optimizer"] = optimizer
        _base_test(search)
