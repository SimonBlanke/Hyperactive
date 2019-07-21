# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

n_iter_0 = 0
n_iter_1 = 100
random_state = 0
cv = 3
n_jobs = 8

search_config = {
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(1, 100, 1),
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": [1],
        "subsample": np.arange(0.05, 1.01, 0.1),
        "max_features": np.arange(0.05, 1.01, 0.1),
    }
}

warm_start = {"sklearn.ensemble.GradientBoostingClassifier": {"n_estimators": [3]}}


def test_HillClimbingOptimizer():
    from hyperactive import HillClimbingOptimizer

    opt0 = HillClimbingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = HillClimbingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_StochasticHillClimbingOptimizer():
    from hyperactive import StochasticHillClimbingOptimizer

    opt0 = StochasticHillClimbingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = StochasticHillClimbingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_TabuOptimizer():
    from hyperactive import TabuOptimizer

    opt0 = TabuOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = TabuOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_RandomSearchOptimizer():
    from hyperactive import RandomSearchOptimizer

    opt0 = RandomSearchOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = RandomSearchOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_RandomRestartHillClimbingOptimizer():
    from hyperactive import RandomRestartHillClimbingOptimizer

    opt0 = RandomRestartHillClimbingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = RandomRestartHillClimbingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_RandomAnnealingOptimizer():
    from hyperactive import RandomAnnealingOptimizer

    opt0 = RandomAnnealingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = RandomAnnealingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_SimulatedAnnealingOptimizer():
    from hyperactive import SimulatedAnnealingOptimizer

    opt0 = SimulatedAnnealingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = SimulatedAnnealingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_StochasticTunnelingOptimizer():
    from hyperactive import StochasticTunnelingOptimizer

    opt0 = StochasticTunnelingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = StochasticTunnelingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


"""


def test_ParallelTemperingOptimizer():
    from hyperactive import ParallelTemperingOptimizer

    opt0 = ParallelTemperingOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=n_jobs
    )
    opt0.fit(X, y)

    opt1 = ParallelTemperingOptimizer(
        search_config, n_iter_1, random_state=random_state, verbosity=0, cv=cv, n_jobs=n_jobs
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


"""


def test_ParticleSwarmOptimizer():
    from hyperactive import ParticleSwarmOptimizer

    opt0 = ParticleSwarmOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = ParticleSwarmOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_EvolutionStrategyOptimizer():
    from hyperactive import EvolutionStrategyOptimizer

    opt0 = EvolutionStrategyOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = EvolutionStrategyOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_BayesianOptimizer():
    from hyperactive import BayesianOptimizer

    opt0 = BayesianOptimizer(
        search_config, n_iter_0, random_state=random_state, verbosity=0, cv=cv, n_jobs=1
    )
    opt0.fit(X, y)

    opt1 = BayesianOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best
