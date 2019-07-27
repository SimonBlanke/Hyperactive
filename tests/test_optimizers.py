# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

n_iter_0 = 100
random_state = 0
cv = 2
n_jobs = 1

search_config = {
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}

warm_start = {"sklearn.tree.DecisionTreeClassifier": {"max_depth": [1]}}


def test_HillClimbingOptimizer():
    from hyperactive import HillClimbingOptimizer

    opt = HillClimbingOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_StochasticHillClimbingOptimizer():
    from hyperactive import StochasticHillClimbingOptimizer

    opt = StochasticHillClimbingOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_TabuOptimizer():
    from hyperactive import TabuOptimizer

    opt = TabuOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_RandomSearchOptimizer():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_RandomRestartHillClimbingOptimizer():
    from hyperactive import RandomRestartHillClimbingOptimizer

    opt = RandomRestartHillClimbingOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_RandomAnnealingOptimizer():
    from hyperactive import RandomAnnealingOptimizer

    opt = RandomAnnealingOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_SimulatedAnnealingOptimizer():
    from hyperactive import SimulatedAnnealingOptimizer

    opt = SimulatedAnnealingOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_StochasticTunnelingOptimizer():
    from hyperactive import StochasticTunnelingOptimizer

    opt = StochasticTunnelingOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_ParallelTemperingOptimizer():
    from hyperactive import ParallelTemperingOptimizer

    opt = ParallelTemperingOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_ParticleSwarmOptimizer():
    from hyperactive import ParticleSwarmOptimizer

    opt = ParticleSwarmOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_EvolutionStrategyOptimizer():
    from hyperactive import EvolutionStrategyOptimizer

    opt = EvolutionStrategyOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)


def test_BayesianOptimizer():
    from hyperactive import BayesianOptimizer

    opt = BayesianOptimizer(
        search_config,
        100,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=n_jobs,
        warm_start=warm_start,
    )
    opt.fit(X, y)
