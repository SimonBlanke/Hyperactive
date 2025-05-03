"""Experiment adapter for sklearn cross-validation experiments."""

from sklearn import clone
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
    scoring : callable or str
        sklearn scoring function or metric to evaluate the model's performance.
    cv : int or cross-validation generator
        The number of folds or cross-validation strategy to be used.
    X : array-like, shape (n_samples, n_features)
            The input data for the model.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values for the model.
    """

    def __init__(self, estimator, scoring, cv, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv

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
            cv=self.cv,
        )

        add_info_d = {
            "score_time": cv_results["score_time"],
            "fit_time": cv_results["fit_time"],
            "n_test_samples": _num_samples(self.X),
        }

        return cv_results["test_score"].mean(), add_info_d
