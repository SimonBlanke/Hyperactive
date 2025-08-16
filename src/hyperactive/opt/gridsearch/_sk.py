"""Grid search optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import ParameterGrid

from hyperactive.base import BaseOptimizer


class GridSearchSk(BaseOptimizer):
    """Grid search optimizer, with backend selection and sklearn style parameter grid.

    Parameters
    ----------
    param_grid : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    error_score : float, default=np.nan
        The score to assign if an error occurs during the evaluation of a parameter set.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

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
    >>> from hyperactive.opt import GridSearchSk as GridSearch
    >>> param_grid = {
    ...     "C": [0.01, 0.1, 1, 10],
    ...     "gamma": [0.0001, 0.01, 0.1, 1, 10],
    ... }
    >>> grid_search = GridSearch(param_grid, experiment=sklearn_exp)

    3. running the grid search:
    >>> best_params = grid_search.solve()

    Best parameters can also be accessed via the attributes:
    >>> best_params = grid_search.best_params_
    """

    def __init__(
        self,
        param_grid=None,
        error_score=np.nan,
        experiment=None,
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

    def _solve(self, experiment, param_grid, error_score):
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
        self.best_score_ = scores[best_index]

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
        from hyperactive.experiment.integrations import SklearnCvExperiment

        sklearn_exp = SklearnCvExperiment.create_test_instance()
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": [0.0001, 0.01, 0.1, 1, 10],
        }
        params_sklearn = {
            "experiment": sklearn_exp,
            "param_grid": param_grid,
        }

        from hyperactive.experiment.toy import Ackley

        ackley_exp = Ackley.create_test_instance()
        param_grid = {
            "x0": np.linspace(-5, 5, 10),
            "x1": np.linspace(-5, 5, 10),
        }
        params_ackley = {
            "experiment": ackley_exp,
            "param_grid": param_grid,
        }

        return [params_sklearn, params_ackley]
