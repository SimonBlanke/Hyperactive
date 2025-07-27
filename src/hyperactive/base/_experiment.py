"""Base class for experiment."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
from skbase.base import BaseObject


class BaseExperiment(BaseObject):
    """Base class for experiment."""

    _tags = {
        "object_type": "experiment",
        "python_dependencies": None,
        "property:randomness": "random",  # random or deterministic
        # if deterministic, two calls of score will result in the same value
        # random = two calls may result in different values; same as "stochastic"
        "property:higher_or_lower_is_better": "higher",  # "higher", "lower", "mixed"
        # whether higher or lower scores are better
    }

    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        """Score parameters, with kwargs call. Same as score call."""
        score, _ = self.score(kwargs)
        return score

    @property
    def __name__(self):
        return type(self).__name__

    def paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str
            The parameter names of the search parameters.
        """
        return self._paramnames()

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str
            The parameter names of the search parameters.
        """
        raise NotImplementedError

    def evaluate(self, params):
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
        paramnames = self.paramnames()
        if not set(params.keys()) <= set(paramnames):
            raise ValueError("Parameters do not match.")
        res, metadata = self._evaluate(params)
        res = np.float64(res)
        return res, metadata

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
        raise NotImplementedError

    def score(self, params):
        """Score the parameters - with sign such that higher is always better.

        Same as ``evaluate`` call except for the sign chosen so that higher is better.

        If the tag ``property:higher_or_lower_is_better`` is set to
        ``"lower"``, the result is ``-self.evaluate(params)``.

        If the tag is set to ``"higher"``, the result is
        identical to ``self.evaluate(params)``.

        Parameters
        ----------
        params : dict with string keys
            Parameters to score.

        Returns
        -------
        float
            The score of the parameters.
        dict
            Additional metadata about the search.
        """
        hib = self.get_tag("property:higher_or_lower_is_better", "lower")
        if hib == "higher":
            sign = 1
        elif hib == "lower":
            sign = -1

        eval_res = self.evaluate(params)
        value = eval_res[0]
        metadata = eval_res[1]

        return sign * value, metadata
