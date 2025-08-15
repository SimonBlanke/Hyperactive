# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sys import platform

from tqdm import tqdm

if platform.startswith("linux"):
    initializer = tqdm.set_lock
    initargs = (tqdm.get_lock(),)
else:
    initializer = None
    initargs = ()


def single_process(process_func, process_infos):
    return [process_func(*info) for info in process_infos]


def multiprocessing_wrapper(process_func, process_infos, n_processes):
    import multiprocessing as mp

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
