"""
Internal helpers that bridge behavioural differences between
scikit-learn versions.  Import *private* scikit-learn symbols **only**
here and nowhere else.

Copyright: Hyperactive contributors
License: MIT
"""

from __future__ import annotations

import warnings
from typing import Dict, Any

import sklearn
from packaging import version

_SK_VERSION = version.parse(sklearn.__version__)


# ------------------------------------------------------------------
# A) Replacement for `_deprecate_Xt_in_inverse_transform`
# ------------------------------------------------------------------
if _SK_VERSION < version.parse("1.7"):
    # Still exists → re-export
    from sklearn.utils.deprecation import _deprecate_Xt_in_inverse_transform
else:
    # Removed in 1.7 → provide drop-in replacement
    def _deprecate_Xt_in_inverse_transform(  # noqa: N802  keep sklearn’s name
        X: Any | None,
        Xt: Any | None,
    ):
        """
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


# ------------------------------------------------------------------
# B) Replacement for `_check_method_params`
#    (still present in 1.7, but could be removed later)
# ------------------------------------------------------------------
try:
    from sklearn.utils.validation import _check_method_params  # noqa: F401
except ImportError:  # fallback for future releases

    def _check_method_params(  # type: ignore[override]  # noqa: N802
        X,
        params: Dict[str, Any],
    ):
        # passthrough – rely on estimator & indexable for validation
        return params


__all__ = [
    "_deprecate_Xt_in_inverse_transform",
    "_check_method_params",
]
