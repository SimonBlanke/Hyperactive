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
    }

    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        """Score parameters, with kwargs call."""
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
