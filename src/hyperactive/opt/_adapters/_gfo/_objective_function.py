# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .dictionary import DictClass


def gfo2hyper(search_space, para):
    values_dict = {}
    for _, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class ObjectiveFunction(DictClass):
    def __init__(self, objective_function):
        super().__init__()

        self.objective_function = objective_function

    def run_callbacks(self, type_):
        if self.callbacks and type_ in self.callbacks:
            [callback(self) for callback in self.callbacks[type_]]

    def convert(self, search_space):
        # wrapper for GFOs
        def _model(para):
            para = gfo2hyper(search_space, para)
            self.para_dict = para

            return self.objective_function(self)

        _model.__name__ = self.objective_function.__name__
        return _model
