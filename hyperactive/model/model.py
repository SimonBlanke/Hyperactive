# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time


class Model:
    def __init__(self, func_, nth_process):
        self.func_ = func_
        self.nth_process = nth_process

    def train_model(self, keras_para_dict, X, y):
        start_time = time.time()
        result = self.func_(keras_para_dict, X, y)
        eval_time = time.time() - start_time

        if isinstance(result, tuple):
            score = result[0]
            model = result[1]
        elif isinstance(result, float):
            score = result
            model = None
        else:
            print("Error: model function must return float or tuple")

        return score, eval_time, model
