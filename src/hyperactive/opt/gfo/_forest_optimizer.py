from hyperactive.opt._adapters._gfo import _BaseGFOadapter


class ForestOptimizer(_BaseGFOadapter):
    """Forest optimizer.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the the search process.
    warm_start_smbo
        The warm start for SMBO.
    max_sample_size : int
        The maximum number of points to sample.
    sampling : dict
        The sampling method to use.
    replacement : bool
        Whether to sample with replacement.
    tree_regressor : str
        The tree regressor model to use.
    tree_para : dict
        The model specific parameters for the tree regressor.
    xi : float
        The xi parameter for the tree regressor.
    n_iter : int, default=100
        The number of iterations to run the optimizer.
    verbose : bool, default=False
        If True, print the progress of the optimization process.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of ForestOptimizer with a scikit-learn experiment:

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

    2. setting up the forestOptimizer optimizer:
    >>> from hyperactive.opt import ForestOptimizer
    >>> import numpy as np
    >>>
    >>> config = {
    ...     "search_space": {
    ...         "C": [0.01, 0.1, 1, 10],
    ...         "gamma": [0.0001, 0.01, 0.1, 1, 10],
    ...     },
    ...     "n_iter": 100,
    ... }
    >>> optimizer = ForestOptimizer(experiment=sklearn_exp, **config)

    3. running the optimization:
    >>> best_params = optimizer.solve()

    Best parameters can also be accessed via:
    >>> best_params = optimizer.best_params_
    """

    _tags = {
        "info:name": "Forest Optimizer",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "explore",
        "info:compute": "middle",
    }

    def __init__(
        self,
        search_space=None,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0.1,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling=None,
        replacement=True,
        tree_regressor="extra_tree",
        tree_para=None,
        xi=0.03,
        n_iter=100,
        verbose=False,
        experiment=None,
    ):
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.replacement = replacement
        self.tree_regressor = tree_regressor
        self.tree_para = tree_para
        self.xi = xi
        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints
        self.n_iter = n_iter
        self.experiment = experiment
        self.verbose = verbose

        super().__init__()

    def _get_gfo_class(self):
        """Get the GFO class to use.

        Returns
        -------
        class
            The GFO class to use. One of the concrete GFO classes
        """
        from gradient_free_optimizers import ForestOptimizer

        return ForestOptimizer

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get the test parameters for the optimizer.

        Returns
        -------
        dict with str keys
            The test parameters dictionary.
        """
        params = super().get_test_params()
        experiment = params[0]["experiment"]
        more_params = {
            "experiment": experiment,
            "replacement": True,
            "tree_para": {"n_estimators": 50},
            "xi": 0.33,
            "search_space": {
                "C": [0.01, 0.1, 1, 10],
                "gamma": [0.0001, 0.01, 0.1, 1, 10],
            },
            "n_iter": 100,
        }
        params.append(more_params)
        return params
