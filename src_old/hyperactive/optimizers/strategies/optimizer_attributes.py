# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd


class OptimizerAttributes:
    def __init__(self):
        self.best_para = None
        self.best_score = None
        self.best_since_iter = None
        self.eval_times = None
        self.iter_times = None
        self.search_data = None
        self.random_seed = None

    def _add_result_attributes(
        self,
        best_para,
        best_score,
        best_since_iter,
        eval_times,
        iter_times,
        search_data,
        random_seed,
    ):
        if self.best_para is None:
            self.best_para = best_para
        else:
            if best_score > self.best_score:
                self.best_para = best_para

        if self.best_score is None:
            self.best_score = best_score
        else:
            if best_score > self.best_score:
                self.best_score = best_score

        if self.best_since_iter is None:
            self.best_since_iter = best_since_iter
        else:
            if best_score > self.best_score:
                self.best_since_iter = best_since_iter

        if self.eval_times is None:
            self.eval_times = eval_times
        else:
            self.eval_times = self.eval_times + eval_times

        if self.iter_times is None:
            self.iter_times = iter_times
        else:
            self.iter_times = self.iter_times + eval_times

        if self.search_data is None:
            self.search_data = search_data
        else:
            self.search_data = pd.concat(
                [self.search_data, search_data], ignore_index=True
            )

        if self.random_seed is None:
            self.random_seed = random_seed
