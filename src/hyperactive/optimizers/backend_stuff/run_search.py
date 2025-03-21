# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .distribution import (
    single_process,
    joblib_wrapper,
    multiprocessing_wrapper,
    pathos_wrapper,
)
from .process import _process_


def proxy(args):
    return _process_(*args)


dist_dict = {
    "joblib": (joblib_wrapper, _process_),
    "multiprocessing": (multiprocessing_wrapper, proxy),
    "pathos": (pathos_wrapper, proxy),
}


def _get_distribution(distribution):
    if hasattr(distribution, "__call__"):
        return (distribution, _process_), {}

    elif isinstance(distribution, dict):
        dist_key = list(distribution.keys())[0]
        dist_paras = list(distribution.values())[0]

        return dist_dict[dist_key], dist_paras

    elif isinstance(distribution, str):
        return dist_dict[distribution], {}


def run_search(searches, distribution, n_processes):
    if n_processes == "auto":
        n_processes = len(searches)

    searches_tuple = [(search,) for search in searches]

    if n_processes == 1:
        results_list = single_process(_process_, searches)
    else:
        (distribution, process_func), dist_paras = _get_distribution(
            distribution
        )

        results_list = distribution(process_func, searches_tuple, n_processes)

    return results_list
