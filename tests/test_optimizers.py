# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target

n_iter_0 = 100
random_state = 0
n_jobs = 1


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=3)

    return scores.mean(), model


search_config = {
    model: {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}


def test_HillClimbingOptimizer():
    opt = Hyperactive(search_config, optimizer="HillClimbing")
    opt.fit(X, y)


def test_StochasticHillClimbingOptimizer():
    opt = Hyperactive(search_config, optimizer="StochasticHillClimbing")
    opt.fit(X, y)


def test_TabuOptimizer():
    opt = Hyperactive(search_config, optimizer="TabuSearch")
    opt.fit(X, y)


def test_RandomSearchOptimizer():
    opt = Hyperactive(search_config, optimizer="RandomSearch")
    opt.fit(X, y)


def test_RandomRestartHillClimbingOptimizer():
    opt = Hyperactive(search_config, optimizer="RandomRestartHillClimbing")
    opt.fit(X, y)


def test_RandomAnnealingOptimizer():
    opt = Hyperactive(search_config, optimizer="RandomAnnealing")
    opt.fit(X, y)


def test_SimulatedAnnealingOptimizer():
    opt = Hyperactive(search_config, optimizer="SimulatedAnnealing")
    opt.fit(X, y)


def test_StochasticTunnelingOptimizer():
    opt = Hyperactive(search_config, optimizer="StochasticTunneling")
    opt.fit(X, y)


def test_ParallelTemperingOptimizer():
    opt = Hyperactive(search_config, optimizer="ParallelTempering")
    opt.fit(X, y)


def test_ParticleSwarmOptimizer():
    opt = Hyperactive(search_config, optimizer="ParticleSwarm")
    opt.fit(X, y)


def test_EvolutionStrategyOptimizer():
    opt = Hyperactive(search_config, optimizer="EvolutionStrategy")
    opt.fit(X, y)


def test_BayesianOptimizer():
    opt = Hyperactive(search_config, optimizer="Bayesian")
    opt.fit(X, y)
