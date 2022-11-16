# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def _process_(nth_process, optimizer):
    optimizer.search(nth_process)

    return {
        "nth_process": nth_process,
        "best_para": optimizer.best_para,
        "best_score": optimizer.best_score,
        "best_iter": optimizer.best_since_iter,
        "eval_times": optimizer.eval_times,
        "iter_times": optimizer.iter_times,
        "positions": optimizer.positions,
        "search_data": optimizer.search_data,
        "memory_values_df": optimizer.memory_values_df,
        "random_seed": optimizer.random_seed,
    }
