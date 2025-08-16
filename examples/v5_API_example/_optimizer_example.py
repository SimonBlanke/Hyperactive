import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor


from hyperactive.search_config import SearchConfig
from hyperactive.optimization.gradient_free_optimizers import (
    HillClimbingOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
)
from hyperactive.optimization.talos import TalosOptimizer

from experiments.sklearn import SklearnExperiment
from experiments.test_function import AckleyFunction


data = load_diabetes()
X, y = data.data, data.target


search_config1 = SearchConfig(
    max_depth=list(np.arange(2, 15, 1)),
    min_samples_split=list(np.arange(2, 25, 2)),
)


TalosOptimizer()

experiment1 = SklearnExperiment()
experiment1.setup(DecisionTreeRegressor, X, y, cv=4)


optimizer = HillClimbingOptimizer()
optimizer.add_search(experiment1, search_config1, n_iter=100)
hyper = optimizer
hyper.run(max_time=5)


