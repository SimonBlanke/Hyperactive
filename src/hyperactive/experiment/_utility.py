def add_callbacks(before=None, after=None):
    def decorator(function):
        def wrapper(self, param):
            if before:
                [before_callback(self, param) for before_callback in before]
            result = function(self, param)
            if after:
                [after_callback(self, param) for after_callback in after]
            return result

        return wrapper

    return decorator
