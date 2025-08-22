"""Internal helpers that bridge behavioural differences between scikit-learn versions.

Import *private* scikit-learn symbols **only** here and nowhere else.

Copyright: Hyperactive contributors
License: MIT
"""

from __future__ import annotations

import warnings
from typing import Any

import sklearn
from packaging import version
from sklearn.utils.validation import indexable

_SK_VERSION = version.parse(sklearn.__version__)


def _safe_validate_X_y(estimator, X, y):
    """
    Version-independent replacement for naive validate_data(X, y).

    • Ensures X is 2-D.
    • Allows y to stay 1-D (required by scikit-learn >=1.7 checks).
    • Uses BaseEstimator._validate_data when available so that
      estimator tags and sample-weight checks keep working.
    """
    X, y = indexable(X, y)

    if hasattr(estimator, "_validate_data"):
        return estimator._validate_data(
            X,
            y,
            validate_separately=(
                {"ensure_2d": True},  # parameters for X
                {"ensure_2d": False},  # parameters for y
            ),
        )

    # Fallback for very old scikit-learn versions (<0.23)
    from sklearn.utils.validation import check_X_y

    return check_X_y(X, y, ensure_2d=True)


def _safe_refit(estimator, X, y, fit_params):
    if estimator.refit:
        estimator._refit(X, y, **fit_params)

        # make the wrapper itself expose n_features_in_
        if hasattr(estimator.best_estimator_, "n_features_in_"):
            estimator.n_features_in_ = estimator.best_estimator_.n_features_in_
    else:
        # Even when `refit=False` we must satisfy the contract
        estimator.n_features_in_ = X.shape[1]


# Replacement for `_deprecate_Xt_in_inverse_transform`
if _SK_VERSION < version.parse("1.7"):
    # Still exists → re-export
    from sklearn.utils.deprecation import _deprecate_Xt_in_inverse_transform
else:
    # Removed in 1.7 - provide drop-in replacement
    def _deprecate_Xt_in_inverse_transform(  # noqa: N802  keep sklearn's name
        X: Any | None,
        Xt: Any | None,
    ):
        """Handle deprecation of Xt parameter in inverse_transform.

        scikit-learn ≤1.6 accepted both the old `Xt` parameter and the new
        `X` parameter for `inverse_transform`.  When only `Xt` is given we
        return `Xt` and raise a deprecation warning (same behaviour that
        scikit-learn had before 1.7); otherwise we return `X`.
        """
        if Xt is not None:
            warnings.warn(
                "'Xt' was deprecated in scikit-learn 1.2 and has been "
                "removed in 1.7; use the positional argument 'X' instead.",
                FutureWarning,
                stacklevel=2,
            )
            return Xt
        return X


# Replacement for `_check_method_params`
try:
    from sklearn.utils.validation import _check_method_params  # noqa: F401
except ImportError:  # fallback for future releases

    def _check_method_params(  # type: ignore[override]  # noqa: N802
        X,
        params: dict[str, Any],
    ):
        # passthrough - rely on estimator & indexable for validation
        return params


__all__ = [
    "_deprecate_Xt_in_inverse_transform",
    "_check_method_params",
]
