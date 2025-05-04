import numpy as np

from hyperactive.base import BaseExperiment


class SphereFunction(BaseExperiment):
    """A simple Sphere function.

    This is a common test function for optimization
    algorithms. The function is defined as the sum of the squares of
    its input parameters plus a constant.

    Parameters
    ----------
    const (float)
        A constant offset added to the sum of squares.
    n_dim : int, optional, default=2
        The number of dimensions for the Sphere function. The default is 2.
    """

    def __init__(self, const, n_dim=2):
        self.const = const
        self.n_dim = n_dim

        super().__init__()

    def _paramnames(self):
        return [f"x{i}" for i in range(self.n_dim)]

    def _score(self, params):
        return np.sum(np.array(params) ** 2) + self.const, {}
