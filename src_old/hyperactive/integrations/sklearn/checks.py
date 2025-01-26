class Checks:
    _fit_successful = False

    def verify_fit(function):
        def wrapper(self, X, y):
            out = function(self, X, y)
            self._fit_successful = True
            return out

        return wrapper
