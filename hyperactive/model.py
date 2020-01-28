# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numbers


def is_numeric(variable):
    return isinstance(variable, numbers.Number)


class Model:
    def __init__(self, func_, nth_process, _main_args_):
        self.func_ = func_
        self.nth_process = nth_process
        self.X = _main_args_.X
        self.y = _main_args_.y

    def train_model(self, para_dict):
        start_time = time.time()
        results = self.func_(para_dict, self.X, self.y)
        eval_time = time.time() - start_time

        if isinstance(results, tuple):
            self.n_results = len(results)

            score = results[0]
            self.rest = results[1]
        else:
            self.n_results = 1
            score = results
            self.rest = None

        if is_numeric(score):
            return score, eval_time
        else:
            print("Error: model function must return numeric variable")
