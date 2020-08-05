# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm import tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed


def multiprocessing_wrapper(n_processes, run_job_parallel):
    n_process_range = range(0, n_processes)

    pool = Pool(n_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    return pool.map(run_job_parallel, n_process_range)


def joblib_wrapper(n_processes, run_job_parallel):
    n_process_range = range(0, n_processes)

    return Parallel(n_jobs=n_processes)(
        delayed(run_job_parallel)(i) for i in n_process_range
    )
