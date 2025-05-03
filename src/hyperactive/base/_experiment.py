"""Base class for experiment."""

from skbase.base import BaseObject


class BaseExperiment(BaseObject):
    """Base class for experiment."""

    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        """Score parameters, with kwargs call."""
        score, _ = self.score(**kwargs)
        return score

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

    def score(self, **params):
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
        return self._score(**params)

    def _score(self, **params):
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
