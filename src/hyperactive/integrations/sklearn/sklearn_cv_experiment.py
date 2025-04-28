"""Experiment adapter for sklearn cross-validation experiments."""

from sklearn import clone
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import _num_samples

from hyperactive.base import BaseExperiment

class SklearnCvExperiment(BaseExperiment):

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

    def _score(self, **params):
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
            self.estimator,
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
