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

    def get_best(self, score, para):
        self.nth_iter += 1

        if score > self.best_score:
            self.best_score = score
            self.best_para = para
            self.best = 1
        else:
            self.best = 0

    def __call__(self, search_space, progress_collector):
        # wrapper for GFOs
        def _model(para):
            para = gfo2hyper(search_space, para)
            self.para_dict = para
            results = self.objective_function(self)

            if progress_collector:
                progress_dict = para

                if isinstance(results, tuple):
                    score = results[0]
                    results_dict = results[1]
                else:
                    score = results
                    results_dict = {}

                # keep track on best score and para
                self.get_best(score, para)

                results_dict["score"] = score

                progress_dict.update(results_dict)
                progress_dict["score_best"] = self.best_score
                progress_dict["nth_iter"] = self.nth_iter
                progress_dict["best"] = self.best

                progress_dict["nth_process"] = self.optimizer.nth_process

                progress_collector.append(progress_dict)

            # ltm save after iteration
            # self.ltm.ltm_obj_func_wrapper(results, para, nth_process)

            return results

        _model.__name__ = self.objective_function.__name__
        return _model
