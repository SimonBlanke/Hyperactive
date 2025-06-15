# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def _process_(nth_process, optimizer):
    if "progress_bar" in optimizer.verbosity:
        from skbase.utils.dependencies import _check_soft_dependencies

        _check_soft_dependencies("tqdm", obj="progress_bar verbosity")

        from tqdm import tqdm

        p_bar = tqdm(
            position=nth_process,
            total=optimizer.n_iter,
            ascii=" ─",
            colour="Yellow",
        )
    else:
        p_bar = None

    optimizer.search(nth_process, p_bar)

    if p_bar:
        p_bar.colour = "GREEN"
        p_bar.refresh()
        p_bar.close()

    return {
        "nth_process": nth_process,
        "best_para": optimizer.best_para,
        "best_score": optimizer.best_score,
        "best_iter": optimizer.best_since_iter,
        "eval_times": optimizer.eval_times,
        "iter_times": optimizer.iter_times,
        "search_data": optimizer.search_data,
        "random_seed": optimizer.random_seed,
    }
