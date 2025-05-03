"""Hill climbing optimizer from gfo."""

from gradient_free_optimizers import HillClimbingOptimizer
from hyperactive.base import BaseOptimizer


class HillClimbing(BaseOptimizer):
    """Hill climbing optimizer.

    Parameters
    ----------
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
    """

    def __init__(
        self,
        random_state=None,
        rand_rest_p=0.1,
        epsilon=0.01,
        distribution="uniform",
        n_neighbours=10,
    ):
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours

        super().__init__()

        self._searches = []
        self._experiments = []

    def add_search(self, experiment, search_config: dict):
        """Add a new optimization search process with specified parameters.

        Parameters
        ---------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_config : dict with str keys
            The search configuration dictionary, keys as below.

        search_config has the following keys:

        search_space : dict[str, list]
            The search space to explore. A dictionary with parameter
            names as keys and a numpy array as values.
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
        self._searches.append(experiment, search_config)
        self._experiments.append(experiment)


    def run(self, max_time=None):
        """Run the optimization search process.

        Parameters
        ----------
        max_time : float
            The maximum time used for the optimization process.
        """
        search_config = self._searches[0]
        experiment = self._experiments[0]

        DEFAULT_initialize = {"grid": 4, "random": 2, "vertices": 4}
        hcopt = HillClimbingOptimizer(
            search_space=search_config.get("search_space", None),
            initialize=search_config.get("initialize", DEFAULT_initialize),
            constraints=search_config.get("constraints", []),
            random_state=self.random_state,
            rand_rest_p=self.rand_rest_p,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
        )

        hcopt.search(
            objective_function=experiment,
            n_iter=search_config.get("n_iter", 100),
            max_time=max_time,
        )
        self.best_params_ = hcopt.best_params()

        return self.best_params_
