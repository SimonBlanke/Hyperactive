# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_digits

data = load_digits()
X = data.data
y = data.target

n_iter_0 = 0
n_iter_1 = 10
random_state = 0
cv = 2
n_jobs = 1

search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": range(0, 5), "batch_size": [500], "verbose": [0]},
    "keras.layers.Dense.1": {
        "units": range(1, 100, 1),
        "activation": ["relu", "tanh", "linear", "sigmoid"],
        "kernel_initializer": ["RandomUniform"],
    },
    "keras.layers.Dense.2": {"units": [1], "activation": ["sigmoid"]},
}

warm_start = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [1], "batch_size": [500], "verbose": [0]},
    "keras.layers.Dense.1": {
        "units": [1],
        "activation": ["linear"],
        "kernel_initializer": ["RandomUniform"],
    },
    "keras.layers.Dense.2": {"units": [1], "activation": ["sigmoid"]},
}


def test_HillClimbingOptimizer():
    from hyperactive import HillClimbingOptimizer

    opt0 = HillClimbingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=1,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = HillClimbingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=1,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


test_HillClimbingOptimizer()
"""
def test_StochasticHillClimbingOptimizer():
    from hyperactive import StochasticHillClimbingOptimizer

    opt0 = StochasticHillClimbingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = StochasticHillClimbingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_TabuOptimizer():
    from hyperactive import TabuOptimizer

    opt0 = TabuOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = TabuOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_RandomSearchOptimizer():
    from hyperactive import RandomSearchOptimizer

    opt0 = RandomSearchOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=1,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = RandomSearchOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=1,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best



def test_RandomRestartHillClimbingOptimizer():
    from hyperactive import RandomRestartHillClimbingOptimizer

    opt0 = RandomRestartHillClimbingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = RandomRestartHillClimbingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_RandomAnnealingOptimizer():
    from hyperactive import RandomAnnealingOptimizer

    opt0 = RandomAnnealingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = RandomAnnealingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_SimulatedAnnealingOptimizer():
    from hyperactive import SimulatedAnnealingOptimizer

    opt0 = SimulatedAnnealingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = SimulatedAnnealingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_StochasticTunnelingOptimizer():
    from hyperactive import StochasticTunnelingOptimizer

    opt0 = StochasticTunnelingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = StochasticTunnelingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_ParallelTemperingOptimizer():
    from hyperactive import ParallelTemperingOptimizer

    opt0 = ParallelTemperingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = ParallelTemperingOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_ParticleSwarmOptimizer():
    from hyperactive import ParticleSwarmOptimizer

    opt0 = ParticleSwarmOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = ParticleSwarmOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_EvolutionStrategyOptimizer():
    from hyperactive import EvolutionStrategyOptimizer

    opt0 = EvolutionStrategyOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = EvolutionStrategyOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best


def test_BayesianOptimizer():
    from hyperactive import BayesianOptimizer

    opt0 = BayesianOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt0.fit(X, y)

    opt1 = BayesianOptimizer(
        search_config,
        n_iter_1,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt1.fit(X, y)

    assert opt0.score_best < opt1.score_best
"""
