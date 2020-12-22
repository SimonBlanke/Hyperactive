# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from multiprocessing import Pool
from joblib import Parallel, delayed


def single_process(process_func, search_processes_infos):
    results = [process_func(**search_processes_infos[0])]

    return results


def multiprocessing_wrapper(process_func, search_processes_paras, **kwargs):
    n_jobs = len(search_processes_paras)

    pool = Pool(n_jobs, **kwargs)
    results = pool.map(process_func, search_processes_paras)

    return results


def joblib_wrapper(process_func, search_processes_paras, **kwargs):
    n_jobs = len(search_processes_paras)

    jobs = [
        delayed(process_func)(**info_dict)
        for info_dict in search_processes_paras
    ]
    results = Parallel(n_jobs=n_jobs, **kwargs)(jobs)

    return results
