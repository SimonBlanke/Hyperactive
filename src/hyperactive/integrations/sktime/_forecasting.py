# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("sktime", severity="none"):
    from sktime.forecasting.base._delegate import _DelegatedForecaster
else:
    from skbase.base import BaseEstimator as _DelegatedForecaster

from hyperactive.experiment.integrations.sktime_forecasting import (
    SktimeForecastingExperiment,
)


class ForecastingOptCV(_DelegatedForecaster):
    """Tune an sktime forecaster via any optimizer in the hyperactive toolbox.

    ``ForecastingOptCV`` uses any available tuning engine from ``hyperactive``
    to tune a forecaster by backtesting.

    It passes backtesting results as scores to the tuning engine,
    which identifies the best hyperparameters.

    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``,
      this results in the same algorithm as ``ForecastingGridSearchCV``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    Configuration of the tuning engine is as per the respective documentation.

    Formally, ``ForecastingOptCV`` does the following:

    In ``fit``:

    * wraps the ``forecaster``, ``scoring``, and other parameters
      into a ``SktimeForecastingExperiment`` instance, which is passed to the optimizer
      ``optimizer`` as the ``experiment`` argument.
    * Optimal parameters are then obtained from ``optimizer.solve``, and set
      as ``best_params_`` and ``best_forecaster_`` attributes.
    *  If ``refit=True``, ``best_forecaster_`` is fitted to the entire ``y`` and ``X``.

    In ``predict`` and ``predict``-like methods, calls the respective method
    of the ``best_forecaster_`` if ``refit=True``.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.

    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.

    cv : sktime BaseSplitter descendant
        determines split of ``y`` and possibly ``X`` into test and train folds
        y is always split according to ``cv``, see above
        if ``cv_X`` is not passed, ``X`` splits are subset to ``loc`` equal to ``y``
        if ``cv_X`` is passed, ``X`` is split according to ``cv_X``

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update

    update_behaviour : str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    cv_X : sktime BaseSplitter descendant, optional
        determines split of ``X`` into test and train folds
        default is ``X`` being split to identical ``loc`` indices as ``y``
        if passed, must have same number of splits as ``cv``

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

    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the ``forecasters_`` attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.

    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the ``forecasters_`` attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

    Example
    -------
    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    For illustration, we use grid search, this can be replaced by any other optimizer.

    1. defining the tuned estimator:
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from hyperactive.integrations.sktime import ForecastingOptCV
    >>> from hyperactive.opt import GridSearchSk as GridSearch
    >>>
    >>> param_grid = {"strategy": ["mean", "last", "drift"]}
    >>> tuned_naive = ForecastingOptCV(
    ...     NaiveForecaster(),
    ...     GridSearch(param_grid),
    ...     cv=ExpandingWindowSplitter(
    ...         initial_window=12, step_length=3, fh=range(1, 13)
    ...     ),
    ... )

    2. fitting the tuned estimator:
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>>
    >>> tuned_naive.fit(y_train, fh=range(1, 13))
    ForecastingOptCV(...)
    >>> y_pred = tuned_naive.predict()

    3. obtaining best parameters and best forecaster
    >>> best_params = tuned_naive.best_params_
    >>> best_forecaster = tuned_naive.best_forecaster_
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "fkiraly",
        "python_dependencies": "sktime",
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "best_forecaster_"

    def __init__(
        self,
        forecaster,
        optimizer,
        cv,
        strategy="refit",
        update_behaviour="full_refit",
        scoring=None,
        refit=True,
        error_score=np.nan,
        cv_X=None,
        backend=None,
        backend_params=None,
        tune_by_instance=False,
        tune_by_variable=False,
    ):
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.cv = cv
        self.strategy = strategy
        self.update_behaviour = update_behaviour
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.cv_X = cv_X
        self.backend = backend
        self.backend_params = backend_params
        self.tune_by_instance = tune_by_instance
        self.tune_by_variable = tune_by_variable
        super().__init__()

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        import time

        from sktime.utils.validation.forecasting import check_scoring

        # Handle broadcasting options
        if self.tune_by_instance or self.tune_by_variable:
            return self._fit_with_broadcasting(y, X, fh)

        forecaster = self.forecaster.clone()

        scoring = check_scoring(self.scoring, obj=self)
        # scoring_name = f"test_{scoring.name}"
        self.scorer_ = scoring

        # Count number of CV splits
        self.n_splits_ = self.cv.get_n_splits(y)

        experiment = SktimeForecastingExperiment(
            forecaster=forecaster,
            scoring=scoring,
            cv=self.cv,
            X=X,
            y=y,
            strategy=self.strategy,
            error_score=self.error_score,
            cv_X=self.cv_X,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        optimizer = self.optimizer.clone()
        optimizer.set_params(experiment=experiment)
        best_params = optimizer.solve()

        self.best_params_ = best_params
        self.best_forecaster_ = forecaster.set_params(**best_params)

        # Store cv_results from optimizer if available
        if hasattr(optimizer, "results"):
            self.cv_results_ = optimizer.results
        else:
            # Create a basic cv_results_ dict
            self.cv_results_ = {"best_params": best_params}

        # Store best_index_ and best_score_ if available from optimizer
        if hasattr(optimizer, "best_score"):
            self.best_score_ = optimizer.best_score
        else:
            # Calculate best score by evaluating best params
            best_score, _ = experiment.score(best_params)
            self.best_score_ = best_score

        self.best_index_ = 0  # For single best result

        # Refit model with best parameters and track time.
        if self.refit:
            start_time = time.time()
            self.best_forecaster_.fit(y=y, X=X, fh=fh)
            end_time = time.time()
            self.refit_time_ = end_time - start_time
        else:
            self.refit_time_ = 0.0

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        if not self.refit:
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True to make predictions,"
                f" but found refit=False. If refit=False, {self.__class__.__name__} can"
                " be used only to tune hyper-parameters, as a parameter estimator."
            )
        return super()._predict(fh=fh, X=X)

    def _fit_with_broadcasting(self, y, X, fh):
        """Fit with broadcasting options (tune_by_instance or tune_by_variable).

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        import pandas as pd
        from sktime.utils.validation.forecasting import check_scoring

        scoring = check_scoring(self.scoring, obj=self)
        self.scorer_ = scoring
        self.n_splits_ = self.cv.get_n_splits(y)

        # Determine if we need to broadcast
        is_panel = "MultiIndex" in str(type(getattr(y, "index", None)))
        is_multivariate = isinstance(y, pd.DataFrame) and len(y.columns) > 1

        forecasters_list = []

        # Handle tune_by_instance for Panel/Hierarchical data
        if self.tune_by_instance and is_panel:
            # Get unique instances
            if hasattr(y.index, "levels"):
                instances = y.index.get_level_values(0).unique()
            else:
                instances = [0]  # Single instance fallback

            for instance in instances:
                # Extract instance data
                if hasattr(y.index, "levels"):
                    y_instance = y.loc[instance]
                    X_instance = X.loc[instance] if X is not None else None
                else:
                    y_instance = y
                    X_instance = X

                # Fit for this instance
                tuner = type(self)(
                    forecaster=self.forecaster.clone(),
                    optimizer=self.optimizer.clone(),
                    cv=self.cv,
                    strategy=self.strategy,
                    update_behaviour=self.update_behaviour,
                    scoring=self.scoring,
                    refit=self.refit,
                    error_score=self.error_score,
                    cv_X=self.cv_X,
                    backend=self.backend,
                    backend_params=self.backend_params,
                    tune_by_instance=False,
                    tune_by_variable=self.tune_by_variable,
                )
                tuner.fit(y_instance, X=X_instance, fh=fh)

                forecasters_list.append(
                    {
                        "instance": instance,
                        "forecaster": tuner.best_forecaster_,
                        "best_params": tuner.best_params_,
                        "best_score": tuner.best_score_,
                    }
                )

            # Store as DataFrame
            self.forecasters_ = pd.DataFrame(forecasters_list)
            # Set a representative best_forecaster_
            self.best_forecaster_ = forecasters_list[0]["forecaster"]
            self.best_params_ = forecasters_list[0]["best_params"]
            self.best_score_ = forecasters_list[0]["best_score"]

        # Handle tune_by_variable for multivariate data
        elif self.tune_by_variable and is_multivariate:
            variables = y.columns

            for variable in variables:
                # Extract variable data
                y_var = y[[variable]]
                X_var = X if X is not None else None

                # Fit for this variable
                tuner = type(self)(
                    forecaster=self.forecaster.clone(),
                    optimizer=self.optimizer.clone(),
                    cv=self.cv,
                    strategy=self.strategy,
                    update_behaviour=self.update_behaviour,
                    scoring=self.scoring,
                    refit=self.refit,
                    error_score=self.error_score,
                    cv_X=self.cv_X,
                    backend=self.backend,
                    backend_params=self.backend_params,
                    tune_by_instance=False,
                    tune_by_variable=False,
                )
                tuner.fit(y_var, X=X_var, fh=fh)

                forecasters_list.append(
                    {
                        "variable": variable,
                        "forecaster": tuner.best_forecaster_,
                        "best_params": tuner.best_params_,
                        "best_score": tuner.best_score_,
                    }
                )

            # Store as DataFrame
            self.forecasters_ = pd.DataFrame(forecasters_list)
            # Set a representative best_forecaster_
            self.best_forecaster_ = forecasters_list[0]["forecaster"]
            self.best_params_ = forecasters_list[0]["best_params"]
            self.best_score_ = forecasters_list[0]["best_score"]
        else:
            # If broadcasting was requested but not applicable, fall back to regular fit
            return self._fit(y, X, fh)

        self.best_index_ = 0
        self.cv_results_ = {"forecasters": self.forecasters_}
        self.refit_time_ = 0.0

        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        update_behaviour = self.update_behaviour

        if update_behaviour == "full_refit":
            super()._update(y=y, X=X, update_params=update_params)
        elif update_behaviour == "inner_only":
            self.best_forecaster_.update(y=y, X=X, update_params=update_params)
        elif update_behaviour == "no_update":
            self.best_forecaster_.update(y=y, X=X, update_params=False)
        else:
            raise ValueError(
                'update_behaviour must be one of "full_refit", "inner_only",'
                f' or "no_update", but found {update_behaviour}'
            )
        return self

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
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import (
            MeanAbsolutePercentageError,
            mean_absolute_percentage_error,
        )
        from sktime.split import SingleWindowSplitter

        from hyperactive.opt.gfo import HillClimbing
        from hyperactive.opt.gridsearch import GridSearchSk
        from hyperactive.opt.random_search import RandomSearchSk

        params_gridsearch = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "optimizer": GridSearchSk(param_grid={"window_length": [2, 5]}),
            "scoring": MeanAbsolutePercentageError(symmetric=True),
        }
        params_randomsearch = {
            "forecaster": PolynomialTrendForecaster(),
            "cv": SingleWindowSplitter(fh=1),
            "optimizer": RandomSearchSk(param_distributions={"degree": [1, 2]}),
            "scoring": mean_absolute_percentage_error,
            "update_behaviour": "inner_only",
        }
        params_hillclimb = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "optimizer": HillClimbing(
                search_space={"window_length": [2, 5]},
                n_iter=10,
                n_neighbours=5,
            ),
            "scoring": "MeanAbsolutePercentageError(symmetric=True)",
            "update_behaviour": "no_update",
        }
        return [params_gridsearch, params_randomsearch, params_hillclimb]
