# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


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
        self.best_para = best_para
        self.best_score = best_score
        self.best_since_iter = best_since_iter
        self.eval_times = eval_times
        self.iter_times = iter_times
        self.search_data = search_data
        self.random_seed = random_seed
