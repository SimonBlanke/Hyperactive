"""2D parabola function, common benchmark for optimization algorithms."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.base import BaseExperiment


class Parabola(BaseExperiment):
    r"""2D parabola, common benchmark for optimization algorithms.

    Parabola parameterized by the formula:

    .. math::
        f(x, y) = a * (x^2 + y^2) + b * x + c * y

    where :math:`a`, :math:`b`, and :math:`c` are coefficients which can
    be set as parameters.

    The function arguments :math:`x` and :math:`y`
    are the input variables of the `score` method,
    and are set as `x` and `y` respectively.

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
        "property:higher_or_lower_is_better": "lower",
        # values are "higher", "lower", "mixed"
        # whether higher or lower scores are better
    }

    def __init__(self, a=1.0, b=0.0, c=0.0):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def _paramnames(self):
        return ["x", "y"]

    def _evaluate(self, params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """
        x = params["x"]
        y = params["y"]

        return self.a * (x**2 + y**2) + self.b * x + self.c * y, {}

    @classmethod
    def _get_score_params(self):
        """Return settings for testing score/evaluate functions. Used in tests only.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        params0 = {"x": 0, "y": 0}
        params1 = {"x": 1, "y": 1}
        return [params0, params1]
