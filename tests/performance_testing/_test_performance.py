# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive

data = load_breast_cancer()
X = data.data
y = data.target

random_state = 1
n_iter_min = 0
n_iter_max = 100


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


search_config = {
    model: {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}

warm_start = {model: {"max_depth": [1]}}


def test_HillClimbing():
    opt0 = Hyperactive(
        search_config,
        optimizer="HillClimbing",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="HillClimbing",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_StochasticHillClimbing():
    opt0 = Hyperactive(
        search_config,
        optimizer="StochasticHillClimbing",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="StochasticHillClimbing",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_TabuOptimizer():
    opt0 = Hyperactive(
        search_config,
        optimizer="TabuSearch",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="TabuSearch",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_RandomSearch():
    opt0 = Hyperactive(
        search_config,
        optimizer="RandomSearch",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="RandomSearch",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_RandomRestartHillClimbing():
    opt0 = Hyperactive(
        search_config,
        optimizer="RandomRestartHillClimbing",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="RandomRestartHillClimbing",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_RandomAnnealing():
    opt0 = Hyperactive(
        search_config,
        optimizer="RandomAnnealing",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="RandomAnnealing",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_SimulatedAnnealing():
    opt0 = Hyperactive(
        search_config,
        optimizer="SimulatedAnnealing",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="SimulatedAnnealing",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_StochasticTunneling():
    opt0 = Hyperactive(
        search_config,
        optimizer="StochasticTunneling",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="StochasticTunneling",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_ParallelTempering():
    opt0 = Hyperactive(
        search_config,
        optimizer="ParallelTempering",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="ParallelTempering",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_ParticleSwarm():
    opt0 = Hyperactive(
        search_config,
        optimizer="ParticleSwarm",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="ParticleSwarm",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_EvolutionStrategy():
    opt0 = Hyperactive(
        search_config,
        optimizer="EvolutionStrategy",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="EvolutionStrategy",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best


def test_Bayesian():
    opt0 = Hyperactive(
        search_config,
        optimizer="Bayesian",
        n_iter=n_iter_min,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt0.search(X, y)

    opt1 = Hyperactive(
        search_config,
        optimizer="Bayesian",
        n_iter=n_iter_max,
        random_state=random_state,
        warm_start=warm_start,
    )
    opt1.search(X, y)

    assert opt0._optimizer_.score_best < opt1._optimizer_.score_best
