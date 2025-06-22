"""Experiment adapter for sklearn cross-validation experiments."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from sklearn import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import _num_samples

from hyperactive.base import BaseExperiment


class SklearnCvExperiment(BaseExperiment):
    """Experiment adapter for sklearn cross-validation experiments.

    This class is used to perform cross-validation experiments using a given
    sklearn estimator. It allows for hyperparameter tuning and evaluation of
    the model's performance using cross-validation.

    The score returned is the mean of the cross-validation scores,
    of applying cross-validation to ``estimator`` with the parameters given in
    ``score`` ``params``.

    The cross-validation performed is specified by the ``cv`` parameter,
    and the scoring metric is specified by the ``scoring`` parameter.
    The ``X`` and ``y`` parameters are the input data and target values,
    which are used in fit/predict cross-validation.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator to be used for the experiment.
    X : array-like, shape (n_samples, n_features)
            The input data for the model.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values for the model.
    cv : int or cross-validation generator, default = KFold(n_splits=3, shuffle=True)
        The number of folds or cross-validation strategy to be used.
        If int, the cross-validation used is KFold(n_splits=cv, shuffle=True).
    scoring : callable or str, default = accuracy_score or mean_squared_error
        sklearn scoring function or metric to evaluate the model's performance.
        Default is determined by the type of estimator:
        ``accuracy_score`` for classifiers, and
        ``mean_squared_error`` for regressors, as per sklearn convention
        through the default ``score`` method of the estimator.

    Example
    -------
    >>> from hyperactive.experiment.integrations import SklearnCvExperiment
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import KFold
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>>
    >>> sklearn_exp = SklearnCvExperiment(
    ...     estimator=SVC(),
    ...     scoring=accuracy_score,
    ...     cv=KFold(n_splits=3, shuffle=True),
    ...     X=X,
    ...     y=y,
    ... )
    >>> params = {"C": 1.0, "kernel": "linear"}
    >>> score, add_info = sklearn_exp.score(params)

    For default choices of ``scoring`` and ``cv``:
    >>> sklearn_exp = SklearnCvExperiment(
    ...     estimator=SVC(),
    ...     X=X,
    ...     y=y,
    ... )
    >>> params = {"C": 1.0, "kernel": "linear"}
    >>> score, add_info = sklearn_exp.score(params)

    """

    def __init__(self, estimator, X, y, scoring=None, cv=None):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv

        super().__init__()

        if cv is None:
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=3, shuffle=True)
        elif isinstance(cv, int):
            from sklearn.model_selection import KFold

            self._cv = KFold(n_splits=cv, shuffle=True)
        else:
            self._cv = cv

        # check if scoring is a scorer by checking for "estimator" in signature
        if scoring is None:
            self._scoring = check_scoring(self.estimator)
        # check using inspect.signature for "estimator" in signature
        elif callable(scoring):
            from inspect import signature

            if "estimator" in signature(scoring).parameters:
                self._scoring = scoring
            else:
                from sklearn.metrics import make_scorer

                self._scoring = make_scorer(scoring)
        self.scorer_ = self._scoring

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str
            The parameter names of the search parameters.
        """
        return list(self.estimator.get_params().keys())

    def _score(self, params):
        """Score the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to score.

        Returns
        -------
        float
            The score of the parameters.
        dict
            Additional metadata about the search.
        """
        estimator = clone(self.estimator)
        estimator.set_params(**params)

        cv_results = cross_validate(
            estimator,
            self.X,
            self.y,
            scoring=self._scoring,
            cv=self._cv,
        )

        add_info_d = {
            "score_time": cv_results["score_time"],
            "fit_time": cv_results["fit_time"],
            "n_test_samples": _num_samples(self.X),
        }

        return cv_results["test_score"].mean(), add_info_d

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
        from sklearn.datasets import load_diabetes, load_iris
        from sklearn.svm import SVC, SVR
        from sklearn.metrics import accuracy_score, mean_absolute_error
        from sklearn.model_selection import KFold

        X, y = load_iris(return_X_y=True)
        params_classif = {
            "estimator": SVC(),
            "scoring": accuracy_score,
            "cv": KFold(n_splits=3, shuffle=True),
            "X": X,
            "y": y,
        }

        X, y = load_diabetes(return_X_y=True)
        params_regress = {
            "estimator": SVR(),
            "scoring": mean_absolute_error,
            "cv": 2,
            "X": X,
            "y": y,
        }

        X, y = load_diabetes(return_X_y=True)
        params_all_default = {
            "estimator": SVR(),
            "X": X,
            "y": y,
        }

        return [params_classif, params_regress, params_all_default]

    @classmethod
    def _get_score_params(self):
        """Return settings for testing the score function. Used in tests only.

        Returns a list, the i-th element corresponds to self.get_test_params()[i].
        It should be a valid call for self.score.

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        score_params_classif = {"C": 1.0, "kernel": "linear"}
        score_params_regress = {"C": 1.0, "kernel": "linear"}
        score_params_defaults = {"C": 1.0, "kernel": "linear"}
        return [score_params_classif, score_params_regress, score_params_defaults]
