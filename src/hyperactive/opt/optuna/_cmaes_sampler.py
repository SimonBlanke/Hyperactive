"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler optimizer."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from ._base_optuna_adapter import _BaseOptunaAdapter


class CmaEsSampler(_BaseOptunaAdapter):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

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
    x0 : dict, default=None
        Initial parameter values for CMA-ES.
    sigma0 : float, default=1.0
        Initial standard deviation for CMA-ES.
    n_startup_trials : int, default=1
        Number of startup trials for CMA-ES.
    experiment : BaseExperiment, optional
        The experiment to optimize parameters for.
        Optional, can be passed later via ``set_params``.

    Examples
    --------
    Basic usage of CmaEsSampler with a scikit-learn experiment:

    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from hyperactive.opt.optuna import CmaEsSampler
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> sklearn_exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)
    >>> param_space = {
    ...     "C": (0.01, 10),
    ...     "gamma": (0.0001, 10),
    ... }
    >>> optimizer = CmaEsSampler(
    ...     param_space=param_space, n_trials=50, experiment=sklearn_exp
    ... )
    >>> best_params = optimizer.run()
    """

    _tags = {
        "info:name": "CMA-ES Sampler",
        "info:local_vs_global": "global",
        "info:explore_vs_exploit": "mixed",
        "info:compute": "high",
        "python_dependencies": ["optuna", "cmaes"],
    }

    def __init__(
        self,
        param_space=None,
        n_trials=100,
        initialize=None,
        random_state=None,
        early_stopping=None,
        max_score=None,
        x0=None,
        sigma0=1.0,
        n_startup_trials=1,
        experiment=None,
    ):
        self.x0 = x0
        self.sigma0 = sigma0
        self.n_startup_trials = n_startup_trials
        
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
        """Get the CMA-ES sampler.

        Returns
        -------
        sampler
            The Optuna CmaEsSampler instance
        """
        import optuna
        
        try:
            import cmaes
        except ImportError:
            raise ImportError(
                "CmaEsSampler requires the 'cmaes' package. "
                "Install it with: pip install cmaes"
            )
        
        sampler_kwargs = {
            "sigma0": self.sigma0,
            "n_startup_trials": self.n_startup_trials,
        }
        
        if self.x0 is not None:
            sampler_kwargs["x0"] = self.x0
            
        if self.random_state is not None:
            sampler_kwargs["seed"] = self.random_state
        
        return optuna.samplers.CmaEsSampler(**sampler_kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the optimizer."""
        params = super().get_test_params(parameter_set)
        params[0].update({
            "sigma0": 0.5,
            "n_startup_trials": 1,
        })
        return params