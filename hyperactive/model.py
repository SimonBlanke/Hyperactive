# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numbers


def is_numeric(variable):
    return isinstance(variable, numbers.Number)


class Model:
    def __init__(self, model, func_para):
        self.model = model
        self.func_para = func_para

    def eval(self, para_dict):
        results_dict = {}

        start_time = time.time()
        results = self.model(
            para_dict, self.func_para["features"], self.func_para["target"]
        )
        eval_time = time.time() - start_time

        results_dict["score"] = results
        results_dict["eval_time"] = eval_time

        if is_numeric(results_dict["score"]):
            return results_dict
        else:
            print("Error: model function must return numeric variable")
