# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def _process_(
    nth_process,
    objective_function,
    optimizer,
    n_iter,
    memory,
    memory_warm_start,
    max_time,
    max_score,
    early_stopping,
    random_state,
    verbosity,
    **kwargs
):
    if "progress_bar" in verbosity:
        verbosity_gfo = ["progress_bar"]
    else:
        verbosity_gfo = []

    optimizer.search(
        objective_function=objective_function,
        n_iter=n_iter,
        max_time=max_time,
        max_score=max_score,
        early_stopping=early_stopping,
        memory=memory,
        memory_warm_start=memory_warm_start,
        verbosity=verbosity_gfo,
        random_state=random_state,
        nth_process=nth_process,
    )

    return {
        "nth_process": nth_process,
        "best_para": optimizer.best_para,
        "best_score": optimizer.best_score,
        "best_iter": optimizer.p_bar._best_since_iter,
        "eval_times": optimizer.eval_time,
        "iter_times": optimizer.iter_time,
        "positions": optimizer.positions,
        "results": optimizer.results,
        "memory_values_df": optimizer.memory_values_df,
    }
