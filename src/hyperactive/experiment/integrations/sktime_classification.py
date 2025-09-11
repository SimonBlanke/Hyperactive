"""Experiment adapter for sktime backtesting experiments."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

from hyperactive.base import BaseExperiment
from hyperactive.experiment.integrations._skl_metrics import (
    _coerce_to_scorer,
    _guess_sign_of_sklmetric,
)


class SktimeClassificationExperiment(BaseExperiment):
    """Experiment adapter for time series classification experiments.

    This class is used to perform cross-validation experiments using a given
    sktime classifier. It allows for hyperparameter tuning and evaluation of
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
    estimator : sktime BaseClassifier descendant (concrete classifier)
        sktime classifier to benchmark

    X : sktime-compatible panel data (Panel scitype)
        Panel data container. Supported formats include:

        - ``pd.DataFrame`` with MultiIndex [instance, time] and variable columns
        - 3D ``np.array`` with shape ``[n_instances, n_dimensions, series_length]``
        - Other formats listed in ``datatypes.SCITYPE_REGISTER``

    y : sktime-compatible tabular data (Table scitype)
        Target variable, typically a 1D ``np.ndarray`` or ``pd.Series``
        of shape ``[n_instances]``.

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
    >>> from hyperactive.experiment.integrations import SktimeClassificationExperiment
    >>> from sklearn.model_selection import KFold
    >>> from sklearn.metrics import accuracy_score
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.classification.dummy import DummyClassifier
    >>>
    >>> X, y = load_unit_test()
    >>>
    >>> sktime_exp = SktimeClassificationExperiment(
    ...     estimator=DummyClassifier(),
    ...     scoring=accuracy_score,
    ...     cv=KFold(n_splits=2),
    ...     X=X,
    ...     y=y,
    ... )
    >>> params = {"strategy": "most_frequent"}
    >>> score, add_info = sktime_exp.score(params)

    For default choices of ``scoring`` and ``cv``:
    >>> sktime_exp = SktimeClassificationExperiment(
    ...     estimator=DummyClassifier(),
    ...     X=X,
    ...     y=y,
    ... )
    >>> params = {"strategy": "most_frequent"}
    >>> score, add_info = sktime_exp.score(params)

    Quick call without metadata return or dictionary:
    >>> score = sktime_exp({"strategy": "most_frequent"})
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "fkiraly",
        "python_dependencies": "sktime",  # python dependencies
    }

    def __init__(
        self,
        estimator,
        X,
        y,
        cv=None,
        scoring=None,
        error_score=np.nan,
        backend=None,
        backend_params=None,
    ):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

        self._scoring = _coerce_to_scorer(scoring, "classifier")

        # Set the sign of the scoring function
        _sign_attr = getattr(self._scoring, "_sign", None)
        if _sign_attr is not None:
            _sign = _sign_attr
        else:
            score_func = getattr(self._scoring, "_score_func", self._scoring)
            _sign = _guess_sign_of_sklmetric(score_func)
        _sign_str = "higher" if _sign == 1 else "lower"
        self.set_tags(**{"property:higher_or_lower_is_better": _sign_str})

        # default handling for cv
        if isinstance(cv, int):
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=cv, shuffle=True)
        elif cv is None:
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=3, shuffle=True)
        else:
            self._cv = cv

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str
            The parameter names of the search parameters.
        """
        return list(self.estimator.get_params().keys())

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
        from sktime.classification.model_evaluation import evaluate

        estimator = self.estimator.clone().set_params(**params)

        # determine metric function for sktime.evaluate via centralized coerce helper
        metric_func = getattr(self._scoring, "_metric_func", None)
        if metric_func is None:
            # very defensive fallback (should not happen due to _coerce_to_scorer)
            from sklearn.metrics import accuracy_score as metric_func  # type: ignore

        results = evaluate(
            estimator,
            cv=self._cv,
            X=self.X,
            y=self.y,
            scoring=metric_func,
            error_score=self.error_score,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        metric = metric_func
        result_name = f"test_{getattr(metric, '__name__', 'score')}"

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
        from sklearn.metrics import brier_score_loss
        from sklearn.model_selection import KFold
        from sktime.classification.dummy import DummyClassifier
        from sktime.datasets import load_unit_test

        X, y = load_unit_test(return_X_y=True, return_type="pd-multiindex")
        params0 = {
            "estimator": DummyClassifier(strategy="most_frequent"),
            "X": X,
            "y": y,
        }

        params1 = {
            "estimator": DummyClassifier(strategy="stratified"),
            "cv": KFold(n_splits=2),
            "X": X,
            "y": y,
            "scoring": brier_score_loss,
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
        val0 = {}
        val1 = {"strategy": "most_frequent"}
        return [val0, val1]
