"""Validation checks for scikit-learn integration."""


class Checks:
    """Checks class."""

    _fit_successful = False

    def verify_fit(function):
        """Verify Fit function."""

        def wrapper(self, X, y):
            """Wrap function call."""
            out = function(self, X, y)
            self._fit_successful = True
            return out

        return wrapper
