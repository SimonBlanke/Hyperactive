"""Validation checks for scikit-learn integration."""

from functools import wraps


class Checks:
    """Checks class."""

    _fit_successful = False

    def verify_fit(function):
        """Mark fit successful and preserve signature."""

        @wraps(function)
        def wrapper(self, *args, **kwargs):
            out = function(self, *args, **kwargs)
            self._fit_successful = True
            return out

        return wrapper
