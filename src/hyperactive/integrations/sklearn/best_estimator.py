"""Best estimator utilities for scikit-learn integration.

Author: Simon Blanke
Email: simon.blanke@yahoo.com
License: MIT License
"""

from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._compat import _deprecate_Xt_in_inverse_transform
from .utils import _estimator_has


# NOTE Implementations of following methods from:
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search.py
# Tag: 1.5.1
class BestEstimator:
    """BestEstimator class."""

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        """Score Samples function."""
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Predict function."""
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict Proba function."""
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict Log Proba function."""
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Decision Function function."""
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        """Transform function."""
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, X=None, Xt=None):
        """Inverse Transform function."""
        X = _deprecate_Xt_in_inverse_transform(X, Xt)
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(X)

    @property
    def classes_(self):
        """Classes  function."""
        _estimator_has("classes_")(self)
        return self.best_estimator_.classes_
