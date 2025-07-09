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
        "property:higher_or_lower_is_better": "lower",  # "higher", "lower", "mixed"
        # whether higher or lower scores are better
    }

    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        """Score parameters, with kwargs call. Same as cost call."""
        score, _ = self.cost(kwargs)
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

    def score(self, params):
        """Score the parameters.

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
        paramnames = self.paramnames()
        if not set(params.keys()) <= set(paramnames):
            raise ValueError("Parameters do not match.")
        res, metadata = self._score(params)
        res = np.float64(res)
        return res, metadata

    def _score(self, params):
        """Score the parameters.

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
        raise NotImplementedError

    def cost(self, params):
        """Score the parameters - with sign such that lower is better.

        Same as ``score`` call except for the sign.

        If the tag ``property:higher_or_lower_is_better`` is set to
        ``"higher"``, the result is ``-self.score(params)``.

        If the tag is set to ``"lower"``, the result is
        identical to ``self.score(params)``.

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
            sign = -1
        elif hib == "lower":
            sign = 1

        score_res = self.score(params)

        return sign * score_res[0], score_res[1]
