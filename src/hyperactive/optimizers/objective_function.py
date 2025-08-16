"""objective_function module for Hyperactive optimization."""

# Email: simon.blanke@yahoo.com
# License: MIT License


from .dictionary import DictClass


def gfo2hyper(search_space, para):
    """Gfo2Hyper function."""
    values_dict = {}
    for _, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class ObjectiveFunction(DictClass):
    """ObjectiveFunction class."""

    def __init__(self, objective_function, optimizer, callbacks, catch, nth_process):
        super().__init__()

        self.objective_function = objective_function
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.catch = catch
        self.nth_process = nth_process

        self.nth_iter = 0

    def run_callbacks(self, type_):
        """Run Callbacks function."""
        if self.callbacks and type_ in self.callbacks:
            [callback(self) for callback in self.callbacks[type_]]

    def __call__(self, search_space):
        """Make object callable with search space."""
        # wrapper for GFOs
        def _model(para):
            self.nth_iter = len(self.optimizer.pos_l)
            para = gfo2hyper(search_space, para)
            self.para_dict = para

            try:
                self.run_callbacks("before")
                results = self.objective_function(self)
                self.run_callbacks("after")
            except tuple(self.catch.keys()) as e:
                results = self.catch[e.__class__]

            return results

        _model.__name__ = self.objective_function.__name__
        return _model
