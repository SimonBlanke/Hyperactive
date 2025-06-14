"""Adapter for sklearn regressors and classifiers for Hyperactive optimizers."""

from sklearn.base import clone
from sklearn.utils.validation import indexable, _check_method_params


class _SklearnAdapter:

    _required_parameters = ["estimator", "optimizer", "params_config"]

    def _refit(self, X, y=None, **fit_params):
        self.best_estimator_ = clone(self.estimator).set_params(
            **clone(self.best_params_, safe=False)
        )

        self.best_estimator_.fit(X, y, **fit_params)
        return self

    def _check_data(self, X, y):
        X, y = indexable(X, y)
        if X is not None:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        if hasattr(self, "_validate_data"):
            validate_data = self._validate_data
        else:
            from sklearn.utils.validation import validate_data

        return validate_data(X, y, ensure_2d=False)

    @property
    def fit_successful(self):
        self._fit_successful
