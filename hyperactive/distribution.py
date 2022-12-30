# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm import tqdm


def single_process(process_func, process_infos):
    results = [process_func(*info) for info in process_infos]

    return results


def multiprocessing_wrapper(process_func, process_infos, n_processes):
    import multiprocessing as mp

    pool = mp.Pool(n_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    results = pool.map(process_func, process_infos)

    return results


def pathos_wrapper(process_func, search_processes_paras, n_processes):
    import pathos.multiprocessing as pmp

    pool = pmp.Pool(n_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    results = pool.map(process_func, search_processes_paras)

    return results


def joblib_wrapper(process_func, search_processes_paras, n_processes):
    from joblib import Parallel, delayed

    jobs = [delayed(process_func)(*info_dict) for info_dict in search_processes_paras]
    results = Parallel(n_jobs=n_processes)(jobs)

    return results
