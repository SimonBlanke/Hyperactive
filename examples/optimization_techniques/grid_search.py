import numpy as np

from hyperactive import Hyperactive
from hyperactive.optimizers import GridSearchOptimizer


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": list(np.arange(-10, 10, 0.1)),
    "y": list(np.arange(-10, 10, 0.1)),
}

opt = GridSearchOptimizer(
    step_size=3,
)


hyper = Hyperactive()
hyper.add_search(sphere_function, search_space, n_iter=1500, optimizer=opt)
hyper.run()
