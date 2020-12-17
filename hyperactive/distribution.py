# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm import tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed

from .process import _process_


def proxy(_dict_):
    return _process_(**_dict_)


def single_process(search_processes_infos):
    results = [_process_(**search_processes_infos[0])]

    return results


def multiprocessing_wrapper(search_processes_infos, **kwargs):
    n_jobs = len(search_processes_infos)
    pool = Pool(
        n_jobs,
        initializer=tqdm.set_lock,
        initargs=(tqdm.get_lock(),),
        **kwargs
    )
    results = pool.map(proxy, search_processes_infos)

    return results


def joblib_wrapper(search_processes_infos, **kwargs):
    n_jobs = len(search_processes_infos)
    jobs = [
        delayed(_process_)(**info_dict) for info_dict in search_processes_infos
    ]
    results = Parallel(n_jobs=n_jobs, **kwargs)(jobs)

    return results
