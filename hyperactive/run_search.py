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


def run_search(opt_pros, distribution, n_processes):
    opts = list(opt_pros.values())

    processes = list(opt_pros.keys())
    optimizers = list(opt_pros.values())
    process_infos = list(zip(processes, optimizers))

    if n_processes == "auto":
        n_processes = len(process_infos)

    if n_processes == 1:
        results_list = single_process(_process_, process_infos)
    else:
        (distribution, process_func), dist_paras = _get_distribution(distribution)

        results_list = distribution(process_func, process_infos, n_processes)

    return results_list
