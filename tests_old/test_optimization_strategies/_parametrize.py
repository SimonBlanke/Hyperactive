from hyperactive.optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    PowellsMethod,
    PatternSearch,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    LipschitzOptimizer,
    DirectAlgorithm,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)


optimizers = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        (DownhillSimplexOptimizer),
        (RandomSearchOptimizer),
        (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (PowellsMethod),
        (PatternSearch),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (SpiralOptimization),
        (EvolutionStrategyOptimizer),
        (BayesianOptimizer),
        (LipschitzOptimizer),
        (DirectAlgorithm),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
    ],
)


optimizers_strat = (
    "Optimizer_strat",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        (DownhillSimplexOptimizer),
        (RandomSearchOptimizer),
        (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (PowellsMethod),
        (PatternSearch),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (SpiralOptimization),
        (EvolutionStrategyOptimizer),
        (BayesianOptimizer),
        (LipschitzOptimizer),
        (DirectAlgorithm),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
    ],
)


optimizers_non_smbo = (
    "Optimizer_non_smbo",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        (DownhillSimplexOptimizer),
        (RandomSearchOptimizer),
        (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (PowellsMethod),
        (PatternSearch),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (SpiralOptimization),
        (EvolutionStrategyOptimizer),
    ],
)


optimizers_smbo = (
    "Optimizer_smbo",
    [
        (BayesianOptimizer),
        (LipschitzOptimizer),
        (DirectAlgorithm),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
    ],
)
