import numpy as np
from hyperactive import Hyperactive


def ackley_function(para):
    x, y = para["x"], para["y"]

    loss = (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.exp(1)
        + 20
    )

    return -loss


search_space = {
    "x": list(np.arange(-10, 10, 0.01)),
    "y": list(np.arange(-10, 10, 0.01)),
}


hyper = Hyperactive(verbosity=False)
hyper.add_search(ackley_function, search_space, n_iter=30)
hyper.run()
