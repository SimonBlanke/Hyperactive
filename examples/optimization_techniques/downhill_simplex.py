import numpy as np

from hyperactive import Hyperactive
from hyperactive.optimizers import DownhillSimplexOptimizer


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": list(np.arange(-10, 10, 0.1)),
    "y": list(np.arange(-10, 10, 0.1)),
}

opt = DownhillSimplexOptimizer(
    alpha=1.2,
    gamma=1.1,
    beta=0.8,
    sigma=1,
)


hyper = Hyperactive()
hyper.add_search(sphere_function, search_space, n_iter=1500, optimizer=opt)
hyper.run()
