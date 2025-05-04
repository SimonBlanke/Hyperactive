"""Hill climbing optimizer from gfo."""

from gradient_free_optimizers import HillClimbingOptimizer
from hyperactive.base import BaseOptimizer


class HillClimbing(BaseOptimizer):
    """Hill climbing optimizer.

    Parameters
    ----------
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later in ``add_search``.
    random_state : None, int, default=None
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float, default=0.1
        The probability of a random iteration during the the search process.
    epsilon : float, default=0.01
        The step-size for the climbing.
    distribution : str, default="uniform"
        The type of distribution to sample from.
    n_neighbours : int, default=10
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
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
    n_iter : int, default=100
        The number of iterations to run the optimizer.
    """

    def __init__(
        self,
        experiment=None,
        random_state=None,
        rand_rest_p=0.1,
        epsilon=0.01,
        distribution="uniform",
        n_neighbours=10,
        search_space=None,
        initialize=None,
        constraints=None,
        n_iter=100,
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

    def _run(self, experiment, **search_config):
        """Run the optimization search process."""
        n_iter = search_config.pop("n_iter", 100)
        max_time = search_config.pop("max_time", None)

        hcopt = HillClimbingOptimizer(**search_config)

        hcopt.search(
            objective_function=experiment,
            n_iter=n_iter,
            max_time=max_time,
        )
        self.best_params_ = hcopt.best_params()

        return self.best_params_
