"""Dynamic experiment to allow passing of functions."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.base import BaseExperiment


class FunctionExperiment(BaseExperiment):
    """Experiment that wraps a function.

    Takes a callable that evaluates parameters; exposes it as the ``evaluate`` method.

    Assumes higher scores are better.

    Parameters
    ----------
    func : callable of signature ``callable(dict) -> float``

    Example
    -------
    >>> def parabola(opt):
    ...     return opt["x"]**2 + opt["y"]**2
    >>> para_exp = FunctionExperiment(parabola)
    >>> params = {"x": 1, "y": 2}
    >>> score, add_info = para_exp.score(params)

    Quick call without metadata return or dictionary:
    >>> score = para_exp(x=1, y=2)
    """  # noqa: E501

    def __init__(self, func):
        self.func = func
        super().__init__()

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
        loss = self.func(params)
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
        params0 = {"func": _func1}
        params1 = {"func": _func2}
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
        params0 = {"x": 0, "y": 0}
        params1 = {"x": 1, "y": 1, "z": 2}
        return [params0, params1]


def _func1(x):
    """Simple function to evaluate parameters."""
    return x["x"]**2 + x["y"]**2


def _func2(x):
    """Another simple function to evaluate parameters."""
    return x["x"]**2 - x["y"]**2 + 10 * x["x"] + 5 * x["z"]
