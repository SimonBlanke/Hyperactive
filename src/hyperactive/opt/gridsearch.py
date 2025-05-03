"""Grid search optimizer."""

from collections.abc import Sequence

import numpy as np

from sklearn.model_selection import ParameterGrid, ParameterSampler, check_cv

from hyperactive.base import BaseOptimizer


class GridSearch(BaseOptimizer):
    """Grid search optimizer.

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

    def __init__(self, error_score=np.nan):
        self.error_score = error_score
        super().__init__()

    def _check_param_grid(self, param_grid):
        """_check_param_grid from sklearn 1.0.2, before it was removed."""
        if hasattr(param_grid, "items"):
            param_grid = [param_grid]

        for p in param_grid:
            for name, v in p.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    raise ValueError("Parameter array should be one-dimensional.")

                if isinstance(v, str) or not isinstance(v, (np.ndarray, Sequence)):
                    raise ValueError(
                        f"Parameter grid for parameter ({name}) needs to"
                        f" be a list or numpy array, but got ({type(v)})."
                        " Single values need to be wrapped in a list"
                        " with one element."
                    )

                if len(v) == 0:
                    raise ValueError(
                        f"Parameter values for parameter ({name}) need "
                        "to be a non-empty sequence."
                    )

    def add_search(self, experiment, search_config: dict):
        """Add a new optimization search process with specified parameters.

        Parameters
        ---------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_config : dict with str keys
            The search configuration dictionary, keys as below.

        search_config has the following keys:

        param_grid : dict[str, list]
            The search space to explore. A dictionary with parameter
            names as keys and a numpy array as values.
        """
        self._searches.append(experiment, search_config)
        self._experiments.append(experiment)

    def run(self, evaluate_candidates):
        """Run the optimization search process."""
        param_grid = self._searches[0]["param_grid"]
        experiment = self._experiments[0]

        self._check_param_grid(param_grid)
        candidate_params = list(ParameterGrid(param_grid))

        scores = [experiment(**candidate_param) for candidate_param in candidate_params]

        best_index = np.argmin(scores)
        best_params = candidate_params[best_index]

        self.best_index_ = best_index
        self.best_params_ = best_params
        self.best_score_ = scores[best_index]

        return best_params
