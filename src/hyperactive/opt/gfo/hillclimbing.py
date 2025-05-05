"""Hill climbing optimizer from gfo."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from gradient_free_optimizers import HillClimbingOptimizer
from hyperactive.base import BaseOptimizer
from skbase.utils.stdout_mute import StdoutMute


class HillClimbing(BaseOptimizer):
    """Hill climbing optimizer.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
        Optional, can be passed later in ``add_search``.
    initialize : dict[str, int], default={"grid": 4, "random": 2, "vertices": 4}
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable], default=[]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int, default=None
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float, default=0.1
        The probability of a random iteration during the the search process.
    epsilon : float, default=0.01
        The step-size for the climbing.
    distribution : str, default="normal"
        The type of distribution to sample from.
    n_neighbours : int, default=10
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    n_iter : int, default=100
        The number of iterations to run the optimizer.
    verbose : bool, default=False
        If True, print the progress of the optimization process.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later in ``add_search``.

    Examples
    --------
    Hill climbing applied to scikit-learn parameter tuning:

    1. defining the experiment to optimize:
    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>>
    >>> sklearn_exp = SklearnCvExperiment(
    ...     estimator=SVC(),
    ...     X=X,
    ...     y=y,
    ... )

    2. setting up the hill climbing optimizer:
    >>> from hyperactive.opt import HillClimbing
    >>> import numpy as np
    >>> 
    >>> hillclimbing_config = {
    ...     "search_space": {
    ...         "C": np.array([0.01, 0.1, 1, 10]),
    ...         "gamma": np.array([0.0001, 0.01, 0.1, 1, 10]),
    ...     },
    ...     "n_iter": 100,
    ... }
    >>> hillclimbing = HillClimbing(experiment=sklearn_exp, **hillclimbing_config)

    3. running the hill climbing search:
    >>> best_params = hillclimbing.run()

    Best parameters can also be accessed via the attributes:
    >>> best_params = hillclimbing.best_params_
    """

    _tags = {
        "python_dependencies": ["gradient-free-optimizers>=1.5.0"],
    }

    def __init__(
        self,
        search_space=None,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0.1,
        epsilon=0.01,
        distribution="normal",
        n_neighbours=10,
        n_iter=100,
        verbose=False,
        experiment=None,
    ):
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints
        self.n_iter = n_iter
        self.experiment = experiment
        self.verbose = verbose

        super().__init__()

        if initialize is None:
            self._initialize = {"grid": 4, "random": 2, "vertices": 4}
        else:
            self._initialize = initialize

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
        """Run the optimization search process."""
        n_iter = search_config.pop("n_iter", 100)
        max_time = search_config.pop("max_time", None)

        hcopt = HillClimbingOptimizer(**search_config)

        with StdoutMute(active=not self.verbose):
            hcopt.search(
                objective_function=experiment.score,
                n_iter=n_iter,
                max_time=max_time,
            )
        self.best_params_ = hcopt.best_para

        return self.best_params_

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
