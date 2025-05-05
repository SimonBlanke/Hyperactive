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

    Example
    -------
    >>> from hyperactive.experiment.toy import AckleyFunction
    >>> ackley = AckleyFunction(A=20)
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
        return [{"A": 0}, {"A": 20}, {"A": -42}]

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
        params0 = {"x0": 0, "x1": 0}
        params1 = {"x0": 1, "x1": 1}
        return [params0, params1]
