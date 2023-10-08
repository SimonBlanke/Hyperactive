import numpy as np

from hyperactive import Hyperactive
from hyperactive.optimizers import StochasticHillClimbingOptimizer


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": list(np.arange(-10, 10, 0.1)),
    "y": list(np.arange(-10, 10, 0.1)),
}

opt = StochasticHillClimbingOptimizer(
    epsilon=0.01,
    n_neighbours=5,
    distribution="laplace",
    p_accept=0.05,
)

hyper = Hyperactive()
hyper.add_search(sphere_function, search_space, n_iter=1500, optimizer=opt)
hyper.run()
