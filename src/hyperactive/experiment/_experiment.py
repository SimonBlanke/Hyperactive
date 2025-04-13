"""Base class for experiment."""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Type
from skbase.base import BaseObject


class BaseExperiment(ABC, BaseObject):
    """Base class for experiment."""

    def __init__(
        self,
        catch: Dict = None,
    ):
        super().__init__()

        self.catch = catch

        self._catch = catch or {}

    def __call__(self, **kwargs):
        """Score parameters, with kwargs call."""
        return self.score(kwargs)

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
        if not set(paramnames) == set(params.keys()):
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

    def backend_adapter(self, backend_adapter, s_space):
        gfo_wrapper_model = backend_adapter(
            experiment=self,
        )

        self.gfo_objective_function = gfo_wrapper_model(s_space())

    @abstractmethod
    def objective_function(self, params):
        pass
