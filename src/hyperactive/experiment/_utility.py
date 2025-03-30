def add_callback(before=None, after=None):
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


def add_catch(catch):
    def decorator(function):
        def wrapper(self, param):
            try:
                result = function(self, param)
            except tuple(catch.keys()) as e:
                result = catch[e.__class__]

            return result

        return wrapper

    return decorator
