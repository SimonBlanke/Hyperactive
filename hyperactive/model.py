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
        results_dict = {}
        start_time = time.time()
        results = self.func_(para_dict, self.X, self.y)
        eval_time = time.time() - start_time

        if isinstance(results, dict):
            if "score" not in results:
                print("Error: model function must return dict with score-keyword")

            results_dict = results
            if "eval_time" not in results_dict:
                results_dict["eval_time"] = eval_time

        else:
            results_dict["score"] = results
            results_dict["eval_time"] = eval_time

        if is_numeric(results_dict["score"]):
            return results_dict
        else:
            print("Error: model function must return numeric variable")
