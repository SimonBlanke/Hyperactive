# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def gfo2hyper(search_space, para):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


def _process_(
    nth_process,
    objective_function,
    search_space,
    optimizer,
    n_iter,
    memory,
    memory_warm_start,
    max_time,
    max_score,
    random_state,
    verbosity,
    **kwargs
):
    def gfo_wrapper_model():
        # wrapper for GFOs
        def _model(para):
            para = gfo2hyper(search_space, para)
            optimizer.para_dict = para
            return objective_function(optimizer)

        _model.__name__ = objective_function.__name__
        return _model

    if "progress_bar" in verbosity:
        verbosity_gfo = ["progress_bar"]
    else:
        verbosity_gfo = []

    optimizer.search(
        objective_function=gfo_wrapper_model(),
        n_iter=n_iter,
        max_time=max_time,
        max_score=max_score,
        memory=memory,
        memory_warm_start=memory_warm_start,
        verbosity=verbosity_gfo,
        random_state=random_state,
        nth_process=nth_process,
    )

    optimizer.print_info(
        verbosity,
        objective_function,
        optimizer.best_score,
        optimizer.best_para,
        optimizer.eval_time,
        optimizer.iter_time,
        n_iter,
    )

    return {
        "nth_process": nth_process,
        "best_para": optimizer.best_para,
        "best_score": optimizer.best_score,
        "positions": optimizer.positions,
        "results": optimizer.results,
    }
