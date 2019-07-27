# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


search_config = {
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}


def test_import_Chimaera():
    from hyperactive import Chimaera

    _ = Chimaera()


def test_import_HillClimbingOptimizer():
    from hyperactive import HillClimbingOptimizer

    _ = HillClimbingOptimizer(search_config, 1)


def test_import_StochasticHillClimbingOptimizer():
    from hyperactive import StochasticHillClimbingOptimizer

    _ = StochasticHillClimbingOptimizer(search_config, 1)


def test_import_TabuOptimizer():
    from hyperactive import TabuOptimizer

    _ = TabuOptimizer(search_config, 1)


def test_import_RandomSearchOptimizer():
    from hyperactive import RandomSearchOptimizer

    _ = RandomSearchOptimizer(search_config, 1)


def test_import_RandomRestartHillClimbingOptimizer():
    from hyperactive import RandomRestartHillClimbingOptimizer

    _ = RandomRestartHillClimbingOptimizer(search_config, 1)


def test_import_RandomAnnealingOptimizer():
    from hyperactive import RandomAnnealingOptimizer

    _ = RandomAnnealingOptimizer(search_config, 1)


def test_import_SimulatedAnnealingOptimizer():
    from hyperactive import SimulatedAnnealingOptimizer

    _ = SimulatedAnnealingOptimizer(search_config, 1)


def test_import_StochasticTunnelingOptimizer():
    from hyperactive import StochasticTunnelingOptimizer

    _ = StochasticTunnelingOptimizer(search_config, 1)


def test_import_ParallelTemperingOptimizer():
    from hyperactive import ParallelTemperingOptimizer

    _ = ParallelTemperingOptimizer(search_config, 1)


def test_import_ParticleSwarmOptimizer():
    from hyperactive import ParticleSwarmOptimizer

    _ = ParticleSwarmOptimizer(search_config, 1)


def test_import_EvolutionStrategyOptimizer():
    from hyperactive import EvolutionStrategyOptimizer

    _ = EvolutionStrategyOptimizer(search_config, 1)


def test_import_BayesianOptimizer():
    from hyperactive import BayesianOptimizer

    _ = BayesianOptimizer(search_config, 1)
