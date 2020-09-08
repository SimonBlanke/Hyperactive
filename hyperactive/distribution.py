# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm import tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed

from .search_process import search_process


def proxy(Dict):
    return search_process(**Dict)


def single_process(search_processes_infos):
    results = [search_process(**search_processes_infos[0])]

    return results


def multiprocessing_wrapper(search_processes_infos):
    n_jobs = len(search_processes_infos)
    pool = Pool(n_jobs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    results = pool.map(proxy, search_processes_infos)

    return results


def joblib_wrapper(search_processes_infos):
    n_jobs = len(search_processes_infos)
    jobs = [delayed(search_process)(**kwargs) for kwargs in search_processes_infos]
    results = Parallel(n_jobs=n_jobs)(jobs)

    return results
