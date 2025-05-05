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
