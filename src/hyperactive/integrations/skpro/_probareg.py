# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from skpro.regression.base import _DelegatedProbaRegressor
else:
    from skbase.base import BaseEstimator as _DelegatedProbaRegressor

from hyperactive.experiment.integrations.skpro_probareg import SkproProbaRegExperiment


class ProbaRegOptCV(_DelegatedProbaRegressor):
    """Tune an sktime classifier via any optimizer in the hyperactive toolbox.

    ``TSCOptCV`` uses any available tuning engine from ``hyperactive``
    to tune a classifier by backtesting.

    It passes backtesting results as scores to the tuning engine,
    which identifies the best hyperparameters.

    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``,
      this results in the same algorithm as ``TSCGridSearchCV``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    Configuration of the tuning engine is as per the respective documentation.

    Formally, ``TSCOptCV`` does the following:

    In ``fit``:

    * wraps the ``estimator``, ``scoring``, and other parameters
      into a ``SktimeClassificationExperiment`` instance, which is passed to the
      optimizer ``optimizer`` as the ``experiment`` argument.
    * Optimal parameters are then obtained from ``optimizer.solve``, and set
      as ``best_params_`` and ``best_estimator_`` attributes.
    *  If ``refit=True``, ``best_estimator_`` is fitted to the entire ``y`` and ``X``.

    In ``predict`` and ``predict``-like methods, calls the respective method
    of the ``best_estimator_`` if ``refit=True``.

    Parameters
    ----------
    estimator : sktime classifier, BaseClassifier instance or interface compatible
        The classifier to tune, must implement the sktime classifier interface.

    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.

    cv : int, sklearn cross-validation generator or an iterable, default=3-fold CV
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None = default = ``KFold(n_splits=3, shuffle=True)``
        - integer, number of folds folds in a ``KFold`` splitter, ``shuffle=True``
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with ``shuffle=False`` so the splits will be the same across calls.

    scoring : str, callable, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set. Can be:

        - a single string resolvable to an sklearn scorer
        - a callable that returns a single value;
        - ``None`` = default = ``accuracy_score``

    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    backend : string, by default "None".
        Parallelization backend to use for runs.
        Runs parallel evaluate if specified and ``strategy="refit"``.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
          but changes the return to (lazy) ``dask.dataframe.DataFrame``.
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    Example
    -------
    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    For illustration, we use grid search, this can be replaced by any other optimizer.

    1. defining the tuned estimator:
    >>> from sktime.classification.dummy import DummyClassifier
    >>> from sklearn.model_selection import KFold
    >>> from hyperactive.integrations.sktime import TSCOptCV
    >>> from hyperactive.opt import GridSearchSk as GridSearch
    >>>
    >>> param_grid = {"strategy": ["most_frequent", "stratified"]}
    >>> tuned_naive = TSCOptCV(
    ...     DummyClassifier(),
    ...     GridSearch(param_grid),
    ...     cv=KFold(n_splits=2, shuffle=False),
    ... )

    2. fitting the tuned estimator:
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(
    ...     return_X_y=True, split="TRAIN", return_type="pd-multiindex"
    ... )
    >>> X_test, _ = load_unit_test(
    ...     return_X_y=True, split="TEST", return_type="pd-multiindex"
    ... )
    >>>
    >>> tuned_naive.fit(X_train, y_train)
    TSCOptCV(...)
    >>> y_pred = tuned_naive.predict(X_test)

    3. obtaining best parameters and best estimator
    >>> best_params = tuned_naive.best_params_
    >>> best_classifier = tuned_naive.best_estimator_
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "fkiraly",
        "python_dependencies": "sktime",
    }

    # attribute for _DelegatedClassifier, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedClassifier docstring
    _delegate_name = "best_estimator_"

    def __init__(
        self,
        estimator,
        optimizer,
        cv=None,
        scoring=None,
        refit=True,
        error_score=np.nan,
        backend=None,
        backend_params=None,
    ):
        self.estimator = estimator
        self.optimizer = optimizer
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params
        super().__init__()

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Changes state to "fitted" = sets is_fitted flag to True

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to

        y : pd.DataFrame, must be same length as X
            labels to fit regressor to

        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis,

            * should have same column name as y, same length as X and y
            * should have entries 0 and 1 (float or int),
              0 = uncensored, 1 = (right) censored

            if None, all observations are assumed to be uncensored.

        Returns
        -------
        self : Reference to self.
        """
        estimator = self.estimator.clone()

        experiment = SkproProbaRegExperiment(
            estimator=estimator,
            scoring=self.scoring,
            cv=self.cv,
            X=X,
            y=y,
            C=C,
            error_score=self.error_score,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        optimizer = self.optimizer.clone()
        optimizer.set_params(experiment=experiment)
        best_params = optimizer.solve()

        self.best_params_ = best_params
        self.best_estimator_ = estimator.set_params(**best_params)

        # Refit model with best parameters.
        if self.refit:
            self.best_estimator_.fit(X=X, y=y, C=C)

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        if not self.refit:
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True to make predictions,"
                f" but found refit=False. If refit=False, {self.__class__.__name__} can"
                " be used only to tune hyper-parameters, as a parameter estimator."
            )
        return super()._predict(X=X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sklearn.model_selection import KFold
        from skpro.metrics import CRPS, ConcordanceHarrell
        from skpro.regression.bootstrap import BootstrapRegressor
        from skpro.regression.residual import ResidualDouble
        from skpro.survival.compose import ConditionUncensored

        from hyperactive.opt.gfo import HillClimbing
        from hyperactive.opt.gridsearch import GridSearchSk
        from hyperactive.opt.random_search import RandomSearchSk

        params_gridsearch = {
            "estimator": ResidualDouble.create_test_instance(),
            "optimizer": GridSearchSk(param_grid={"distr_type": ["Normal", "Laplace"]}),
        }
        param_randomsearch = {
            "estimator": ResidualDouble.create_test_instance(),
            "optimizer": RandomSearchSk(
                param_distributions={"distr_type": ["Normal", "Laplace"]},
                n_iter=2,
            ),
            "cv": 2,
            "scoring": CRPS(),
        }
        params_hillclimb = {
            "estimator": ConditionUncensored(BootstrapRegressor.create_test_instance()),
            "cv": KFold(n_splits=2),
            "optimizer": HillClimbing(
                search_space={"estimator__n_bootstrap_samples": [3, 7, 12]},
                n_iter=10,
                n_neighbours=5,
            ),
            "scoring": ConcordanceHarrell(),
        }

        return [params_gridsearch, param_randomsearch, params_hillclimb]
