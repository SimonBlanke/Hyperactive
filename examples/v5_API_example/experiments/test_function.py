import numpy as np

from hyperactive import BaseExperiment


class SphereFunction(BaseExperiment):
    """
    A class representing a Sphere function experiment.

    This class inherits from BaseExperiment and implements a simple
    Sphere function, which is a common test function for optimization
    algorithms. The function is defined as the sum of the squares of
    its input parameters plus a constant.

    Attributes:
        const (float): A constant added to the function's result.
        n_dim (int): The number of dimensions for the input parameters.

    Methods:
        _score(**params): Computes the Sphere function value for the
        given parameters.
    """

    def __init__(self, const, n_dim=2):
        super().__init__()

        self.const = const
        self.n_dim = n_dim

    def _score(self, **params):
        return np.sum(np.array(params) ** 2) + self.const


class AckleyFunction(BaseExperiment):
    """
    A class representing the Ackley function, used as a benchmark for optimization algorithms.

    Attributes:
        A (float): A constant used in the calculation of the Ackley function.

    Methods:
        _score(**params): Computes the Ackley function value for given parameters 'x0' and 'x1'.

    The Ackley function is a non-convex function used to test optimization algorithms.
    """

    def __init__(self, A):
        super().__init__()
        self.A = A

    def _score(self, **params):
        x = params["x0"]
        y = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        return -(loss1 + loss2 + loss3 + loss4)
