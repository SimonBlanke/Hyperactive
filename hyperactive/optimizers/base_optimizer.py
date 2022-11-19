# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class BaseOptimizer:
    def __init__(self):
        pass

    def _add_result_attributes(
        self,
        best_para,
        best_score,
        best_since_iter,
        eval_times,
        iter_times,
        positions,
        search_data,
        memory_values_df,
        random_seed,
    ):
        self.best_para = best_para
        self.best_score = best_score
        self.best_since_iter = best_since_iter
        self.eval_times = eval_times
        self.iter_times = iter_times
        self.positions = positions
        self.search_data = search_data
        self.memory_values_df = memory_values_df
        self.random_seed = random_seed
