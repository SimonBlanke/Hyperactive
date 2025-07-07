"""Optuna optimizer interface."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from hyperactive.base import BaseOptimizer


class OptunaOptimizer(BaseOptimizer):
    """Optuna optimizer interface.

    Parameters
    ----------
    param_space : dict[str, tuple or list or optuna distributions]
        The search space to explore. Dictionary with parameter names
        as keys and either tuples/lists of (low, high) or
        optuna distribution objects as values.
    n_trials : int, default=100
        Number of optimization trials.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Example
    -------
    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.una import OptunaOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": (0.01, 10),
    ...     "gamma": (0.0001, 10),
    ... }
    >>> optimizer = OptunaOptimizer(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.run()
    """

    _tags = {
        "python_dependencies": ["optuna"],
        "info:name": "Optuna-based optimizer",
    }

    def __init__(
        self,
        param_space=None,
        n_trials=100,
        experiment=None
    ):
        self.param_space = param_space
        self.n_trials = n_trials
        self.experiment = experiment
        super().__init__()

    def _objective(self, trial):
        params = {}
        for key, space in self.param_space.items():
            if hasattr(space, "suggest"):  # optuna distribution object
                params[key] = trial._suggest(space, key)
            elif isinstance(space, (tuple, list)) and len(space) == 2:
                low, high = space
                # Decide type based on low/high type
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = trial.suggest_int(key, low, high)
                else:
                    params[key] = trial.suggest_float(key, low, high, log=False)
            else:
                raise ValueError(f"Invalid parameter space for key '{key}': {space}")

        # Evaluate experiment with suggested params
        return self.experiment(**params)

    def _run(self, experiment, param_space, n_trials):
        import optuna

        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)

        self.best_score_ = study.best_value
        self.best_params_ = study.best_params
        return study.best_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""

        from hyperactive.experiment.integrations import SklearnCvExperiment
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        X, y = load_iris(return_X_y=True)
        sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

        param_space = {
            "C": (0.01, 10),
            "gamma": (0.0001, 10),
        }

        return [{
            "param_space": param_space,
            "n_trials": 10,
            "experiment": sklearn_exp,
        }]
