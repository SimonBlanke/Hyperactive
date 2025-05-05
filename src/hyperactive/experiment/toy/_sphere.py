import numpy as np

from hyperactive.base import BaseExperiment


class SphereFunction(BaseExperiment):
    """A simple Sphere function.

    This is a common test function for optimization
    algorithms. The function is defined as the sum of the squares of
    its input parameters plus a constant.

    Parameters
    ----------
    const : float, optional, default=0
        A constant offset added to the sum of squares.
    n_dim : int, optional, default=2
        The number of dimensions for the Sphere function. The default is 2.
    """

    def __init__(self, const=0, n_dim=2):
        self.const = const
        self.n_dim = n_dim

        super().__init__()

    def _paramnames(self):
        return [f"x{i}" for i in range(self.n_dim)]

    def _score(self, params):
        return np.sum(np.array(params) ** 2) + self.const, {}

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
        params0 = {}
        params1 = {"ndim": 3, "const": 1.0}
        return [params0, params1]

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
        score_params0 = {"x0": 0, "x1": 0}
        score_params1 = {"x0": 1, "x1": 2, "x2": 3}
        return [score_params0, score_params1]
