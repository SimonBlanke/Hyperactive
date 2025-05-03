"""2D parabola."""

from hyperactive.base import BaseExperiment


class Parabola(BaseExperiment):
    """2D parabola.

    Parameters
    ----------
    a : float, default=1.0
        Coefficient of the parabola.
    b : float, default=0.0
        Coefficient of the parabola.
    c : float, default=0.0
        Coefficient of the parabola.
    """

    def __init__(self, a=1.0, b=0.0, c=0.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def _paramnames(self):
        return ["x", "y"]

    def _score(self, params):
        x = params["x"]
        y = params["y"]

        return self.a * (x**2 + y**2) + self.b * x + self.c * y, {}
