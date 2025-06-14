"""Adapter for gfo package."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.base import BaseOptimizer
from skbase.utils.stdout_mute import StdoutMute

__all__ = ["_BaseGFOadapter"]


class _BaseGFOadapter(BaseOptimizer):
    """Adapter base class for gradient-free-optimizers.

    * default tag setting
    * default _run method
    * default get_search_config
    * default get_test_params
    * Handles defaults for "initialize" parameter
    * extension interface: _get_gfo_class, docstring, tags
    """

    _tags = {
        "authors": "SimonBlanke",
        "python_dependencies": ["gradient-free-optimizers>=1.5.0"],
    }

    def __init__(self):

        super().__init__()

        if self.initialize is None:
            self._initialize = {"grid": 4, "random": 2, "vertices": 4}
        else:
            self._initialize = self.initialize

    def _get_gfo_class(self):
        """Get the GFO class to use.

        Returns
        -------
        class
            The GFO class to use. One of the concrete GFO classes
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass."
        )

    def get_search_config(self):
        """Get the search configuration.

        Returns
        -------
        dict with str keys
            The search configuration dictionary.
        """
        search_config = super().get_search_config()
        search_config["initialize"] = self._initialize
        del search_config["verbose"]
        return search_config

    def _run(self, experiment, **search_config):
        """Run the optimization search process.
        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_config : dict with str keys
            identical to return of ``get_search_config``.
        Returns
        -------
        dict with str keys
            The best parameters found during the search.
            Must have keys a subset or identical to experiment.paramnames().
        """
        n_iter = search_config.pop("n_iter", 100)
        max_time = search_config.pop("max_time", None)

        gfo_cls = self._get_gfo_class()
        hcopt = gfo_cls(**search_config)

        with StdoutMute(active=not self.verbose):
            hcopt.search(
                objective_function=experiment.score,
                n_iter=n_iter,
                max_time=max_time,
            )
        best_params = hcopt.best_para
        return best_params

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
        import numpy as np
        from hyperactive.experiment.integrations import SklearnCvExperiment

        sklearn_exp = SklearnCvExperiment.create_test_instance()
        params_sklearn = {
            "experiment": sklearn_exp,
            "search_space": {
                "C": np.array([0.01, 0.1, 1, 10]),
                "gamma": np.array([0.0001, 0.01, 0.1, 1, 10]),
            },
            "n_iter": 100,
        }

        from hyperactive.experiment.toy import Ackley

        ackley_exp = Ackley.create_test_instance()
        params_ackley = {
            "experiment": ackley_exp,
            "search_space": {
                "x0": np.linspace(-5, 5, 10),
                "x1": np.linspace(-5, 5, 10),
            },
            "n_iter": 100,
        }
        
        return [params_sklearn, params_ackley]
