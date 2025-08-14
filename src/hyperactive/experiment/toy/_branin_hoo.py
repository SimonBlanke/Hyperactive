"""Branin-Hoo function, common benchmark for optimization algorithms."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.base import BaseExperiment


class BraninHoo(BaseExperiment):
    r"""Branin-Hoo function, common benchmark for optimization algorithms.

    The Branin-Hoo function is a non-convex function used to test optimization
    algorithms. It is considered on the square domain
    :math:`x \in [-5, 10], y \in [0, 15]`.

    It has three minima with :math:`f(x^*) = 0.397887` at :math:`x* = (-\pi, 12.275)`,
    :math:`(\pi, 2.275)`, and :math:`(9.42478, 2.475)`.

    The Branin-Hoo function is defined as:

    .. math::

        f(x, y) = a (y - b x^2 + c x - d)^2 + e (1 - f) \cos(x) + e

    In this function, the six constants can be set as parameters,
    with default values as defined below.

    Parameters
    ----------
    a : float, default=1.0
        Coefficient of the quartic term.
    b : float, default=5.1 / (4 * np.pi ** 2)
        Coefficient of the quadratic term.
    c : float, default=5 / np.pi
        Coefficient of the linear term.
    d : float, default=6.0
        Constant insie the quartic term.
    e : float, default=10.0
        Coefficient of the cosine term and the constant term.
    f : float, default=1 / (8 * np.pi)
        Coefficient of the cosine term.

    References
    ----------
    [1] Branin, F.H. (1972). "Widely convergent method for finding
        multiple solutions of simultaneous nonlinear equations." IBM Journal of
        Research and Development, 16(1), 504-522.

    Example
    -------
    >>> from hyperactive.experiment.toy import BraninHoo
    >>> branin_hoo = BraninHoo()
    >>> params = {"x": 1.0, "y": 2.0}
    >>> score, add_info = branin_hoo.score(params)

    Quick call without metadata return or dictionary:
    >>> score = branin_hoo(x=1.0, y=2.0)
    """  # noqa: E501

    _tags = {
        "property:randomness": "deterministic",  # random or deterministic
        # if deterministic, two calls of score will result in the same value
        # random = two calls may result in different values; same as "stochastic"
        "property:higher_or_lower_is_better": "lower",
        # values are "higher", "lower", "mixed"
        # whether higher or lower scores are better
    }

    def __init__(
        self,
        a=1.0,
        b=5.1 / (4 * np.pi**2),
        c=5 / np.pi,
        d=6.0,
        e=10.0,
        f=1 / (8 * np.pi),
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

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
        x, y = params["x"], params["y"]

        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f

        res = a * (y - b * x**2 + c * x - d) ** 2 + e * (1 - f) * np.cos(x)
        return res, {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        ``get_test_params`` is a unified interface point to store
        parameter settings for testing purposes. This function is also
        used in ``create_test_instance`` and ``create_test_instances_and_names``
        to construct test instances.

        ``get_test_params`` should return a single ``dict``, or a ``list`` of ``dict``.

        Each ``dict`` is a parameter configuration for testing,
        and can be used to construct an "interesting" test instance.
        A call to ``cls(**params)`` should
        be valid for all dictionaries ``params`` in the return of ``get_test_params``.

        The ``get_test_params`` need not return fixed lists of dictionaries,
        it can also return dynamic or stochastic parameter settings.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # default parameters
        params0 = {}
        # parameters different from the default
        params1 = {"a": 2.0, "b": 6.0, "c": 4.0, "d": 8.0, "e": 12.0, "f": 1 / 4}
        return [params0, params1]

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
        params0 = {"x": 0.0, "y": 0.0}
        params1 = {"x": 1.0, "y": 2.0}
        return [params0, params1]
