# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pathos.multiprocessing as mp
from joblib import Parallel, delayed


def single_process(process_func, process_infos):
    results = [process_func(*info) for info in process_infos]

    return results


def multiprocessing_wrapper(process_func, process_infos, n_processes):
    pool = mp.Pool(n_processes)
    results = pool.map(process_func, process_infos)

    return results


def joblib_wrapper(process_func, process_infos, n_processes):
    jobs = [delayed(process_func)(*info_dict) for info_dict in process_infos]
    results = Parallel(n_jobs=n_processes)(jobs)

    return results
