class Checks:
    def __init__(self) -> None:
        self.fit_successful = False

    def is_fit_successful(function):
        def wrapper(self, *args, **kwargs):
            out = function(self, *args, **kwargs)
            self.fit_successful = True
            return out

        return wrapper
