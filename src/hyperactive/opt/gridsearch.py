"""Grid search optimizer."""

from collections.abc import Sequence

import numpy as np

from sklearn.model_selection import ParameterGrid, ParameterSampler, check_cv

from hyperactive.base import BaseOptimizer


class GridSearch(BaseOptimizer):
    """Grid search optimizer.

    Parameters
    ----------
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later in ``add_search``.
    param_grid : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    error_score : float, default=np.nan
        The score to assign if an error occurs during the evaluation of a parameter set.

    Example
    -------
    Grid search applied to scikit-learn parameter tuning:

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

    2. setting up the grid search optimizer:
    >>> from hyperactive.opt import GridSearch
    >>> param_grid = {
    ...     "C": [0.01, 0.1, 1, 10],
    ...     "gamma": [0.0001, 0.01, 0.1, 1, 10],
    ... }
    ... grid_search = GridSearch(sklearn_exp, param_grid=param_grid)

    3. running the grid search:
    >>> best_params = grid_search.run()

    Best parameters can also be accessed via the attributes:
    >>> best_params = grid_search.best_params_
    """

    def __init__(
        self,
        experiment=None,
        param_grid=None,
        error_score=np.nan,
    ):
        self.experiment = experiment
        self.param_grid = param_grid
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

    def _run(self, experiment, param_grid, error_score):
        """Run the optimization search process."""
        self._check_param_grid(param_grid)
        candidate_params = list(ParameterGrid(param_grid))

        scores = []
        for candidate_param in candidate_params:
            try:
                score = experiment(**candidate_param)
            except Exception:  # noqa: B904
                # Catch all exceptions and assign error_score
                score = error_score
            scores.append(score)

        best_index = np.argmin(scores)
        best_params = candidate_params[best_index]

        self.best_index_ = best_index
        self.best_params_ = best_params
        self.best_score_ = scores[best_index]

        return best_params
