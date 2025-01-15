import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from hyperactive.optimizers import (
    HillClimbingOptimizer,
    RandomRestartHillClimbingOptimizer,
)

from .experiments.test_function import SklearnExperiment
from .search_space_optional import SearchSpace


data = load_diabetes()
X, y = data.data, data.target


search_space = {
    "max_depth": list(np.arange(2, 15, 1)),
    "min_samples_split": list(np.arange(2, 25, 2)),
}

""" optional way of defining search-space
search_space = SearchSpace(
    max_depth=list(np.arange(2, 15, 1)),
    min_samples_split=list(np.arange(2, 25, 2)),
)
"""

experiment = SklearnExperiment(DecisionTreeRegressor, X, y, cv=4)

optimizer1 = HillClimbingOptimizer(n_iter=50)
optimizer2 = RandomRestartHillClimbingOptimizer(n_iter=50, n_jobs=2)

optimizer1.add_search(experiment, search_space)
optimizer2.add_search(experiment, search_space)

# not sure about this way of combining optimizers. Might not be intuitive what the plus means.
hyper = optimizer1 + optimizer2

hyper.run(max_time=5)
