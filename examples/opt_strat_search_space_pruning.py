import numpy as np

from hyperactive import Hyperactive


from hyperactive.optimizers.strategies import CustomOptimizationStrategy
from hyperactive.optimizers import (
    HillClimbingOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
)


opt_strat = CustomOptimizationStrategy()
opt_strat.add_optimizer(RandomSearchOptimizer(), duration=0.5)
opt_strat.prune_search_space()
opt_strat.add_optimizer(HillClimbingOptimizer(), duration=0.5)


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score, {"additional stuff": 1}


search_space = {"x1": list(np.arange(-100, 101, 1))}
n_iter = 100
optimizer = opt_strat

hyper = Hyperactive()
hyper.add_search(
    objective_function,
    search_space,
    n_iter=n_iter,
    n_jobs=1,
    optimizer=optimizer,
    # random_state=1,
)
hyper.run()
