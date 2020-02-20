# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target

memory = False

n_iter = 100


def sphere_function(para, X_train, y_train):
    loss = []
    for key in para.keys():
        if key == "iteration":
            continue
        loss.append(para[key] * para[key])

    return -np.array(loss).sum()


search_config = {
    sphere_function: {"x1": np.arange(-10, 10, 0.1), "x2": np.arange(-10, 10, 0.1)}
}


def test_HillClimbingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="HillClimbing")


def test_StochasticHillClimbingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="StochasticHillClimbing")


def test_TabuOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="TabuSearch")


def test_RandomSearchOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomSearch")


def test_RandomRestartHillClimbingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomRestartHillClimbing")


def test_RandomAnnealingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomAnnealing")


def test_SimulatedAnnealingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="SimulatedAnnealing")


def test_StochasticTunnelingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="StochasticTunneling")


def test_ParallelTemperingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="ParallelTempering")


def test_ParticleSwarmOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="ParticleSwarm")


def test_EvolutionStrategyOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="EvolutionStrategy")


def test_BayesianOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=int(n_iter / 10), optimizer="Bayesian")
