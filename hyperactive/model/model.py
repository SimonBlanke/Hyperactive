# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numpy as np


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
            score = results[0]
            model = results[1]
        elif (
            isinstance(results, float)
            or isinstance(results, np.float64)
            or isinstance(results, np.float32)
        ):
            score = results
            model = None
        else:
            print("Error: model function must return float or tuple")

        return score, eval_time, model
