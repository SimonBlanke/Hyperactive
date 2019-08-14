# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


search_config = {"sklearn.tree.DecisionTreeClassifier": {"max_depth": range(1, 21)}}


def test_import_and_inits():

    from hyperactive import Hydra, Insight, Iota

    _ = Hydra()
    # _ = Insight()
    # _ = Iota()

    from hyperactive import (
        HillClimbingOptimizer,
        StochasticHillClimbingOptimizer,
        TabuOptimizer,
        RandomSearchOptimizer,
        RandomRestartHillClimbingOptimizer,
        RandomAnnealingOptimizer,
        SimulatedAnnealingOptimizer,
        StochasticTunnelingOptimizer,
        ParallelTemperingOptimizer,
        ParticleSwarmOptimizer,
        EvolutionStrategyOptimizer,
        BayesianOptimizer,
    )

    _ = HillClimbingOptimizer(search_config, 1)
    _ = StochasticHillClimbingOptimizer(search_config, 1)
    _ = TabuOptimizer(search_config, 1)
    _ = RandomSearchOptimizer(search_config, 1)
    _ = RandomRestartHillClimbingOptimizer(search_config, 1)
    _ = RandomAnnealingOptimizer(search_config, 1)
    _ = SimulatedAnnealingOptimizer(search_config, 1)
    _ = StochasticTunnelingOptimizer(search_config, 1)
    _ = ParallelTemperingOptimizer(search_config, 1)
    _ = ParticleSwarmOptimizer(search_config, 1)
    _ = EvolutionStrategyOptimizer(search_config, 1)
    _ = BayesianOptimizer(search_config, 1)
