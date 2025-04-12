# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sys import platform
from tqdm import tqdm

from ._process import _process_


if platform.startswith("linux"):
    initializer = tqdm.set_lock
    initargs = (tqdm.get_lock(),)
else:
    initializer = None
    initargs = ()


def proxy(args):
    return _process_(*args)


def single_process(process_func, process_infos):
    return [process_func(info) for info in process_infos]


def multiprocessing_wrapper(process_func, process_infos, n_processes):
    import multiprocessing as mp

    process_infos = tuple(process_infos)

    print("\n process_infos ", process_infos)

    with mp.Pool(n_processes, initializer=initializer, initargs=initargs) as pool:
        return pool.map(process_func, process_infos)


def pathos_wrapper(process_func, search_processes_paras, n_processes):
    import pathos.multiprocessing as pmp

    with pmp.Pool(n_processes, initializer=initializer, initargs=initargs) as pool:
        return pool.map(process_func, search_processes_paras)


def joblib_wrapper(process_func, search_processes_paras, n_processes):
    from joblib import Parallel, delayed

    jobs = [delayed(process_func)(*info_dict) for info_dict in search_processes_paras]
    return Parallel(n_jobs=n_processes)(jobs)


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
        (distribution, process_func), dist_paras = _get_distribution(distribution)

        results_list = distribution(process_func, searches_tuple, n_processes)

    return results_list
