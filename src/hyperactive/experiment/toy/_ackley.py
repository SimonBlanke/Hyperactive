import numpy as np

from hyperactive.base import BaseExperiment


class AckleyFunction(BaseExperiment):
    r"""Ackley function, common benchmark for optimization algorithms.

    The Ackley function is a non-convex function used to test optimization algorithms.
    It is defined as:
    .. math::
        f(x, y) = -A \cdot \exp(-0.2 \sqrt{0.5 (x^2 + y^2)}) - \exp(0.5 (\cos(2 \pi x) + \cos(2 \pi y))) + \exp(1) + A

    where A is a constant.
    Parameters
    ----------
    A : float
        Amplitude constant used in the calculation of the Ackley function.
    """  # noqa: E501

    def __init__(self, A):
        self.A = A
        super().__init__()

    def _paramnames(self):
        return ["x0", "x1"]

    def _score(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        return -(loss1 + loss2 + loss3 + loss4), {}
