"""Hartmann function, common benchmark for optimization algorithms."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.base import BaseExperiment


class Hartmann(BaseExperiment):
    r"""Hartmann function, common benchmark for optimization algorithms.

    The Hartmann function is a non-convex function used to test optimization algorithms,
    typically considered on the unit hypercube.

    It has six local minima and one global minimum :math:`f(x^*) = -3.32237` at
    :math:`x^* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573)`.

    The Hartmann function is defined as:

    .. math::

        f(\mathbf{x}) = - \sum_{i=1}^{4} \alpha_i
        \exp \left( - \sum_{j=1}^{6} A_{ij} (x_j - P_{ij})^2 \right),
        \quad \text{where}

    .. math::

        \alpha = (1.0, \; 1.2, \; 3.0, \; 3.2)

    .. math::

        A =
        \begin{pmatrix}
        10 & 3 & 17 & 3.50 & 1.7 & 8 \\
        0.05 & 10 & 17 & 0.1 & 8 & 14 \\
        3 & 3.5 & 1.7 & 10 & 17 & 8 \\
        17 & 8 & 0.05 & 10 & 0.1 & 14
        \end{pmatrix}

    .. math::

        P = 10^{-4} \times
        \begin{pmatrix}
        1312 & 1696 & 5569 & 124 & 8283 & 5886 \\
        2329 & 4135 & 8307 & 3736 & 1004 & 9991 \\
        2348 & 1451 & 3522 & 2883 & 3047 & 6650 \\
        4047 & 8828 & 8732 & 5743 & 1091 & 381
        \end{pmatrix}

    In this function, the constants can be set as parameters,
    and default to the above values and dimensions if not specified.
    The customizable constants are:

    * ``alpha``: an :math:`m`-vector of coefficients :math:`\alpha`
    * ``A``: a :math:`(m \times n)` shape matrix :math:`A`
    * ``P``: a :math:`(m \times n)` position matrix :math:`P`

    The components of the function argument :math:`x`
    are the input variables of the `score` method,
    and are set as `x0`, `x1`, ..., `x[n-1]` respectively.

    Parameters
    ----------
    alpha : 1D array-like of length m, optional, default=[1.0, 1.2, 3.0, 3.2]
        Coefficients of the Hartmann function.
    A : 2D array-like of shape (m, n), optional, default = as defined above
        Shape matrix of the Hartmann function.
    P : 2D array-like of shape (m, n), optional, default = as defined above
        Position matrix of the Hartmann function.

    References
    ----------
    [1] Hartmann, J.L. (1972). "Some Experiments in Global Optimization".
        Naval Postgraduate School, Monterey, CA.

    Example
    -------
    >>> from hyperactive.experiment.toy import Hartmann
    >>> hartmann = Hartmann()
    >>> params = {"x0": 0.1, "x1": 0.2, "x2": 0.3, "x3": 0.4, "x4": 0.5, "x5": 0.6}
    >>> score, add_info = hartmann.score(params)

    Quick call without metadata return or dictionary:
    >>> score = hartmann(x0=0.1, x1=0.2, x2=0.3, x3=0.4, x4=0.5, x5=0.6)
    """  # noqa: E501

    _tags = {
        "property:randomness": "deterministic",  # random or deterministic
        # if deterministic, two calls of score will result in the same value
        # random = two calls may result in different values; same as "stochastic"
        "property:higher_or_lower_is_better": "lower",
        # values are "higher", "lower", "mixed"
        # whether higher or lower scores are better
    }

    def __init__(self, alpha=None, A=None, P=None):
        self.alpha = alpha
        self.A = A
        self.P = P

        super().__init__()

        if self.alpha is None:
            self._alpha = np.array([1.0, 1.2, 3.0, 3.2])
        else:
            self._alpha = np.asarray(alpha)

        if self.A is None:
            self._A = np.array(
                [
                    [10, 3, 17, 3.50, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ]
            )
        else:
            self._A = np.asarray(A)

        if self.P is None:
            self._P = 1e-4 * np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )
        else:
            self._P = np.asarray(P)

    def _paramnames(self):
        n = self._A.shape[1]  # number of dimensions
        return [f"x{i}" for i in range(n)]

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
        x = np.array([params[xi] for xi in self.paramnames()])

        alpha = self._alpha
        A = self._A
        P = self._P

        res = -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P) ** 2, axis=1)))
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
        params1 = {
            "alpha": np.array([4.0, 3.2, 2.0, 1.2]),
            "A": np.array(
                [
                    [42, 3, 17, 3.50, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 33, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ]
            ),
            "P": np.array(
                [
                    [1312, 196, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 100, 9991],
                    [2348, 1451, 322, 2883, 3047, 6650],
                    [4047, 8828, 8732, 743, 1091, 381],
                ]
            ),
        }
        # different parameters with dimensions 2 x 3
        params2 = {
            "alpha": np.array([1.0, 2.0]),
            "A": np.array([[10, 3, 17], [0.05, 10, 17]]),
            "P": np.array([[1312, 1696, 5569], [2329, 4135, 8307]]),
        }
        return [params0, params1, params2]

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
        params = {"x0": 0.1, "x1": 0.2, "x2": 0.3, "x3": 0.4, "x4": 0.5, "x5": 0.6}
        params2 = {"x0": 0.1, "x1": 0.2, "x2": 0.3}
        return [params, params, params2]
