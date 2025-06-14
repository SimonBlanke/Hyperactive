from hyperactive.opt._adapters._gfo import _BaseGFOadapter


class ParallelTempering(_BaseGFOadapter):
    """Parallel tempering optimizer.

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
    epsilon : float
        The step-size for the climbing.
    distribution : str
        The type of distribution to sample from.
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    n_iter : int, default=100
        The number of iterations to run the optimizer.
    verbose : bool, default=False
        If True, print the progress of the optimization process.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.
    """

    def __init__(
        self,
        search_space=None,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0.1,
        population: int = 5,
        n_iter_swap: int = 5,
        n_iter=100,
        verbose=False,
        experiment=None,
    ):
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.population = population
        self.n_iter_swap = n_iter_swap
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
        from gradient_free_optimizers import ParallelTemperingOptimizer

        return ParallelTemperingOptimizer

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get the test parameters for the optimizer.

        Returns
        -------
        dict with str keys
            The test parameters dictionary.
        """
        import numpy as np

        params = super().get_test_params()
        experiment = params[0]["experiment"]
        more_params = {
            "experiment": experiment,
            "population": 10,
            "n_iter_swap": 3,
            "search_space": {
                "C": np.array([0.01, 0.1, 1, 10]),
                "gamma": np.array([0.0001, 0.01, 0.1, 1, 10]),
            },
            "n_iter": 100,
        }
        params.append(more_params)
        return params
