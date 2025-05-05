"""2D parabola."""

from hyperactive.base import BaseExperiment


class Parabola(BaseExperiment):
    """2D parabola, common benchmark for optimization algorithms.

    Parabola parameterized by the formula:

    .. math::
        f(x, y) = a * (x^2 + y^2) + b * x + c * y

    where :math:`a`, :math:`b`, and :math:`c` are coefficients which can
    be set as parameters.

    Parameters
    ----------
    a : float, default=1.0
        Coefficient of the parabola.
    b : float, default=0.0
        Coefficient of the parabola.
    c : float, default=0.0
        Coefficient of the parabola.

    Example
    -------
    >>> from hyperactive.experiment.toy import Parabola
    >>> parabola = Parabola(a=1.0, b=0.0, c=0.0)
    >>> params = {"x": 1, "y": 2}
    >>> score, add_info = parabola.score(params)
    Quick call without metadata return or dictionary:
    >>> score = parabola(x=1, y=2)
    """

    _tags = {
        "property:randomness": "deterministic",  # random or deterministic
        # if deterministic, two calls of score will result in the same value
        # random = two calls may result in different values; same as "stochastic"
    }

    def __init__(self, a=1.0, b=0.0, c=0.0):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def _paramnames(self):
        return ["x", "y"]

    def _score(self, params):
        x = params["x"]
        y = params["y"]

        return self.a * (x**2 + y**2) + self.b * x + self.c * y, {}

    @classmethod
    def _get_score_params(self):
        """Return settings for the score function.

        Returns a list, the i-th element corresponds to self.get_test_params()[i].
        It should be a valid call for self.score.

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        params0 = {"x": 0, "y": 0}
        params1 = {"x": 1, "y": 1}
        return [params0, params1]
