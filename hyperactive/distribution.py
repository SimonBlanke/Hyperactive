# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import multiprocessing as mp
import pathos.multiprocessing as pmp
from joblib import Parallel, delayed


def single_process(process_func, search_processes_infos):
    results = [process_func(**info) for info in search_processes_infos]

    return results


def multiprocessing_wrapper(
    process_func, search_processes_paras, n_processes, **kwargs
):
    pool = mp.Pool(n_processes, **kwargs)
    results = pool.map(process_func, search_processes_paras)

    return results


def pathos_wrapper(process_func, search_processes_paras, n_processes, **kwargs):
    pool = pmp.Pool(n_processes, **kwargs)
    results = pool.map(process_func, search_processes_paras)

    return results


def joblib_wrapper(process_func, search_processes_paras, n_processes, **kwargs):
    jobs = [delayed(process_func)(**info_dict) for info_dict in search_processes_paras]
    results = Parallel(n_jobs=n_processes, **kwargs)(jobs)

    return results
