"""Grid sampler optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from ._base_optuna_adapter import _BaseOptunaAdapter


class GridSampler(_BaseOptunaAdapter):
    """Grid search sampler optimizer.

    Parameters
    ----------
    param_space : dict[str, tuple or list or optuna distributions]
        The search space to explore. Dictionary with parameter names
        as keys and either tuples/lists of (low, high) or
        optuna distribution objects as values.
    n_trials : int, default=100
        Number of optimization trials.
    initialize : dict[str, int], default=None
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    random_state : None, int, default=None
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    early_stopping : int, default=None
        Number of trials after which to stop if no improvement.
    max_score : float, default=None
        Maximum score threshold. Stop optimization when reached.
    search_space : dict, default=None
        Explicit search space for grid search.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of GridSampler with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import GridSampler
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": [0.01, 0.1, 1, 10],
    ...     "gamma": [0.0001, 0.01, 0.1, 1],
    ... }
    >>> optimizer = GridSampler(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.run()
    """

    _tags = {
        "info:name": "Grid Sampler",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "explore",
        "info:compute": "low",
        "python_dependencies": ["optuna"],
    }

    def __init__(
        self,
        param_space=None,
        n_trials=100,
        initialize=None,
        random_state=None,
        early_stopping=None,
        max_score=None,
        search_space=None,
        experiment=None,
    ):
        self.search_space = search_space
        
        super().__init__(
            param_space=param_space,
            n_trials=n_trials,
            initialize=initialize,
            random_state=random_state,
            early_stopping=early_stopping,
            max_score=max_score,
            experiment=experiment,
        )

    def _get_sampler(self):
        """Get the Grid sampler.

        Returns
        -------
        sampler
            The Optuna GridSampler instance
        """
        import optuna
        
        # Convert param_space to Optuna search space format if needed
        search_space = self.search_space
        if search_space is None and self.param_space is not None:
            search_space = {}
            for key, space in self.param_space.items():
                if isinstance(space, list):
                    search_space[key] = space
                elif isinstance(space, (tuple,)) and len(space) == 2:
                    # Convert range to discrete list for grid search
                    low, high = space
                    if isinstance(low, int) and isinstance(high, int):
                        search_space[key] = list(range(low, high + 1))
                    else:
                        # Create a reasonable grid for continuous spaces
                        import numpy as np
                        search_space[key] = np.linspace(low, high, 10).tolist()
        
        return optuna.samplers.GridSampler(search_space)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        from hyperactive.experiment.integrations import SklearnCvExperiment
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        X, y = load_iris(return_X_y=True)
        sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

        param_space = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": [0.0001, 0.01, 0.1, 1],
        }

        return [{
            "param_space": param_space,
            "n_trials": 10,
            "experiment": sklearn_exp,
        }]