# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from .dictionary import DictClass


def gfo2hyper(search_space, para):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class ObjectiveFunction(DictClass):
    def __init__(self, objective_function, optimizer, nth_process):
        super().__init__()

        self.objective_function = objective_function
        self.optimizer = optimizer
        self.nth_process = nth_process

        self.best = 0
        self.nth_iter = 0
        self.best_para = None
        self.best_score = -np.inf

    def __call__(self, search_space):
        # wrapper for GFOs
        def _model(para):
            para = gfo2hyper(search_space, para)
            self.para_dict = para
            results = self.objective_function(self)

            return results

        _model.__name__ = self.objective_function.__name__
        return _model
