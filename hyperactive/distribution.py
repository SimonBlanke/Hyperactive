# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import multiprocessing as mp


try:
    import pathos.multiprocessing as pmp
except ImportError:
    pass

try:
    from joblib import Parallel, delayed
except ImportError:
    pass

try:
    import ray
    from ray.util.multiprocessing import Pool
except ImportError:
    pass


def single_process(process_func, process_infos):
    results = [process_func(*info) for info in process_infos]

    return results


def multiprocessing_wrapper(process_func, process_infos, n_processes):
    pool = mp.Pool(n_processes)
    results = pool.map(process_func, process_infos)

    return results


def pathos_wrapper(process_func, search_processes_paras, n_processes, **kwargs):
    pool = pmp.Pool(n_processes, **kwargs)
    results = pool.map(process_func, search_processes_paras)

    return results


def joblib_wrapper(process_func, search_processes_paras, n_processes, **kwargs):
    jobs = [delayed(process_func)(*info_dict) for info_dict in search_processes_paras]
    results = Parallel(n_jobs=n_processes, **kwargs)(jobs)

    return results


def ray_wrapper(process_func, process_infos, n_processes, **kwargs):
    # ray.init(log_to_driver=False)
    pool = Pool(n_processes)
    results = pool.map(process_func, process_infos)

    return results
