"""Ackley function, common benchmark for optimization algorithms."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.base import BaseExperiment


class Ackley(BaseExperiment):
    r"""Ackley function, common benchmark for optimization algorithms.

    The Ackley function is a non-convex function used to test optimization algorithms.
    It is defined as:

    .. math::
        f(x) = -a \cdot \exp(-\frac{b}{\sqrt{d}\left\|x\right\|}) - \exp(\frac{1}{d} \sum_{i=1}^d\cos (c x_i) ) + a + \exp(1)

    where :math:`a` (= `a`), :math:`b` (= `b`), and :math:`c` (= `c`) are constants,
    :math:`d` (= `d`) is the number of dimensions of the real input vector :math:`x`,
    and :math:`\left\|x\right\|` is the Euclidean norm of the vector :math:`x`.

    The components of the function argument :math:`x`
    are the input variables of the `score` method,
    and are set as `x0`, `x1`, ..., `x[d]` respectively.

    Parameters
    ----------
    a : float, optional, default=20
        Amplitude constant used in the calculation of the Ackley function.
    b : float, optional, default=0.2
        Decay constant used in the calculation of the Ackley function.
    c : float, optional, default=2*pi
        Frequency constant used in the calculation of the Ackley function.
    d : int, optional, default=2
        Number of dimensions for the Ackley function. The default is 2.

    Example
    -------
    >>> from hyperactive.experiment.toy import Ackley
    >>> ackley = Ackley(a=20)
    >>> params = {"x0": 1, "x1": 2}
    >>> score, add_info = ackley.score(params)

    Quick call without metadata return or dictionary:
    >>> score = ackley(x0=1, x1=2)
    """  # noqa: E501

    _tags = {
        "property:randomness": "deterministic",  # random or deterministic
        # if deterministic, two calls of score will result in the same value
        # random = two calls may result in different values; same as "stochastic"
    }

    def __init__(self, a=20, b=0.2, c=2 * np.pi, d=2):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        super().__init__()

    def _paramnames(self):
        return [f"x{i}" for i in range(self.d)]

    def _score(self, params):
        x_vec = np.array([params[f"x{i}"] for i in range(self.d)])

        loss1 = -self.a * np.exp(-self.b * np.sqrt(np.sum(x_vec**2) / self.d))
        loss2 = -np.exp(np.sum(np.cos(self.c * x_vec)) / self.d)
        loss3 = np.exp(1)
        loss4 = self.a

        loss = loss1 + loss2 + loss3 + loss4

        return loss, {}

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
        return [{"a": 0}, {"a": 20, "d": 42}, {"a": -42, "b": 0.5, "c": 1, "d": 10}]

    @classmethod
    def _get_score_params(self):
        """Return settings for testing the score function. Used in tests only.

        Returns a list, the i-th element corresponds to self.get_test_params()[i].
        It should be a valid call for self.score.

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        params0 = {"x0": 0, "x1": 0}
        params1 = {f"x{i}": i + 3 for i in range(42)}
        params2 = {f"x{i}": i**2 for i in range(10)}
        return [params0, params1, params2]
