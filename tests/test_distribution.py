import numpy as np
from tqdm import tqdm
from hyperactive import Hyperactive


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": list(np.arange(-100, 101, 1)),
}


def test_n_jobs_0():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_jobs_1():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.run()

    assert len(hyper.results_list) == 4


def test_n_jobs_2():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=8)
    hyper.run()

    assert len(hyper.results_list) == 8


def test_n_jobs_3():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15)
    hyper.run()


def test_n_jobs_5():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)

    hyper.run()

    assert len(hyper.results_list) == 4


def test_n_jobs_6():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)

    hyper.run()

    assert len(hyper.results_list) == 8


def test_n_jobs_7():
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=-1)
    hyper.run()


def test_multiprocessing_0():
    hyper = Hyperactive(distribution="multiprocessing")
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()


def test_multiprocessing_1():
    hyper = Hyperactive(
        distribution={
            "multiprocessing": {
                "initializer": tqdm.set_lock,
                "initargs": (tqdm.get_lock(),),
            }
        },
    )
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()


def test_joblib_0():
    hyper = Hyperactive(distribution="joblib")
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()


def test_joblib_1():
    from joblib import Parallel, delayed

    def joblib_wrapper(process_func, search_processes_paras, n_jobs, **kwargs):
        n_jobs = len(search_processes_paras)

        jobs = [
            delayed(process_func)(*info_dict) for info_dict in search_processes_paras
        ]
        results = Parallel(n_jobs=n_jobs, *kwargs)(jobs)

        return results

    hyper = Hyperactive(distribution=joblib_wrapper)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)

    hyper.run()


def test_pathos_0():
    hyper = Hyperactive(distribution="pathos")
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()


def test_n_processes_0():
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_processes_1():
    hyper = Hyperactive(n_processes=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_processes_2():
    hyper = Hyperactive(n_processes=4)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_processes_3():
    hyper = Hyperactive(n_processes=4)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=3)
    hyper.run()

    assert len(hyper.results_list) == 3


def test_n_processes_4():
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.run()

    assert len(hyper.results_list) == 4


def test_n_processes_5():
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.run()

    assert len(hyper.results_list) == 8
