"""Base class for optimizer."""

from skbase.base import BaseObject


class BaseOptimizer(BaseObject):
    """Base class for optimizer."""

    def __init__(self):
        super().__init__()

    def add_search(self, experiment, search_config: dict):
        """Add a new optimization search process with specified parameters.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_config : dict with str keys
            The search configuration dictionary.
        """
        self.experiment = experiment
        self.search_config = search_config

    def run(self, max_time=None):
        """Run the optimization search process.

        Parameters
        ----------
        max_time : float
            The maximum time used for the optimization process.
        """
        raise NotImplementedError
