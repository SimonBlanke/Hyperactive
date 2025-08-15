"""Experiment adapter for sktime backtesting experiments."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.base import BaseExperiment


class SktimeForecastingExperiment(BaseExperiment):
    """Experiment adapter for time backtesting experiments.

    This class is used to perform backtesting experiments using a given
    sktime forecaster. It allows for hyperparameter tuning and evaluation of
    the model's performance.

    The score returned is the summary backtesting score,
    of applying ``sktime`` ``evaluate`` to ``estimator`` with the parameters given in
    ``score`` ``params``.

    The backtesting performed is specified by the ``cv`` parameter,
    and the scoring metric is specified by the ``scoring`` parameter.
    The ``X`` and ``y`` parameters are the input data and target values,
    which are used in fit/predict cross-validation.

    Parameters
    ----------
    forecaster : sktime BaseForecaster descendant (concrete forecaster)
        sktime forecaster to benchmark

    cv : sktime BaseSplitter descendant
        determines split of ``y`` and possibly ``X`` into test and train folds
        y is always split according to ``cv``, see above
        if ``cv_X`` is not passed, ``X`` splits are subset to ``loc`` equal to ``y``
        if ``cv_X`` is passed, ``X`` is split according to ``cv_X``

    y : sktime time series container
        Target (endogeneous) time series used in the evaluation experiment

    X : sktime time series container, of same mtype as y
        Exogenous time series used in the evaluation experiment

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update

    scoring : subclass of sktime.performance_metrics.BaseMetric,
        default=None. Used to get a score function that takes y_pred and y_test
        arguments and accept y_train as keyword argument.
        If None, then uses scoring = MeanAbsolutePercentageError(symmetric=True).

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

    Example
    -------
    >>> from hyperactive.experiment.integrations import SktimeForecastingExperiment
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
    >>> from sktime.split import ExpandingWindowSplitter
    >>>
    >>> y = load_airline()
    >>>
    >>> sktime_exp = SktimeForecastingExperiment(
    ...     forecaster=NaiveForecaster(strategy="last"),
    ...     scoring=MeanAbsolutePercentageError(),
    ...     cv=ExpandingWindowSplitter(initial_window=36, step_length=12, fh=12),
    ...     y=y,
    ... )
    >>> params = {"strategy": "mean"}
    >>> score, add_info = sktime_exp.score(params)

    For default choices of ``scoring``:
    >>> sktime_exp = SktimeForecastingExperiment(
    ...     forecaster=NaiveForecaster(strategy="last"),
    ...     cv=ExpandingWindowSplitter(initial_window=36, step_length=12, fh=12),
    ...     y=y,
    ... )
    >>> params = {"strategy": "mean"}
    >>> score, add_info = sktime_exp.score(params)

    Quick call without metadata return or dictionary:
    >>> score = sktime_exp(strategy="mean")
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "fkiraly",
        "python_dependencies": "sktime",  # python dependencies
    }

    def __init__(
        self,
        forecaster,
        cv,
        y,
        X=None,
        strategy="refit",
        scoring=None,
        error_score=np.nan,
        cv_X=None,
        backend=None,
        backend_params=None,
    ):
        self.forecaster = forecaster
        self.X = X
        self.y = y
        self.strategy = strategy
        self.scoring = scoring
        self.cv = cv
        self.error_score = error_score
        self.cv_X = cv_X
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

        if scoring is None:
            from sktime.performance_metrics.forecasting import (
                MeanAbsolutePercentageError,
            )

            self._scoring = MeanAbsolutePercentageError(symmetric=True)
        else:
            self._scoring = scoring

        if scoring is None or scoring.get_tag("lower_is_better", False):
            higher_or_lower_better = "lower"
        else:
            higher_or_lower_better = "higher"
        self.set_tags(**{"property:higher_or_lower_is_better": higher_or_lower_better})

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str
            The parameter names of the search parameters.
        """
        return list(self.forecaster.get_params().keys())

    def _evaluate(self, params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """
        from sktime.forecasting.model_evaluation import evaluate

        results = evaluate(
            self.forecaster,
            cv=self.cv,
            y=self.y,
            X=self.X,
            strategy=self.strategy,
            scoring=self._scoring,
            error_score=self.error_score,
            cv_X=self.cv_X,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        result_name = f"test_{self._scoring.name}"

        res_float = results[result_name].mean()

        return res_float, {"results": results}

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
        from sktime.datasets import load_airline, load_longley
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.split import ExpandingWindowSplitter

        y = load_airline()
        params0 = {
            "forecaster": NaiveForecaster(strategy="last"),
            "cv": ExpandingWindowSplitter(initial_window=36, step_length=12, fh=12),
            "y": y,
        }

        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

        y, X = load_longley()
        params1 = {
            "forecaster": NaiveForecaster(strategy="last"),
            "cv": ExpandingWindowSplitter(initial_window=3, step_length=3, fh=1),
            "y": y,
            "X": X,
            "scoring": MeanAbsolutePercentageError(symmetric=False),
        }

        return [params0, params1]

    @classmethod
    def _get_score_params(self):
        """Return settings for testing score/evaluate functions. Used in tests only.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        val0 = {"strategy": "mean"}
        val1 = {"strategy": "last"}
        return [val0, val1]
