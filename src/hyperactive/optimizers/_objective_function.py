# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ._dictionary import DictClass


def gfo2hyper(search_space, para):
    values_dict = {}
    for _, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class ObjectiveFunction(DictClass):
    def __init__(self, experiment):
        super().__init__()

        self.objective_function = experiment.objective_function
        self.catch = experiment._catch

        self.nth_iter = -1

    def __call__(self, search_space):
        # wrapper for GFOs
        def _model(para):
            self.nth_iter += 1
            para = gfo2hyper(search_space, para)
            self.para_dict = para

            try:
                results = self.objective_function(self)
            except tuple(self.catch.keys()) as e:
                results = self.catch[e.__class__]

            return results

        _model.__name__ = self.objective_function.__name__
        return _model
