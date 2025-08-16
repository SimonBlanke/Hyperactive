"""Test module for distribution functionality."""

import sys

import numpy as np
import pytest
from tqdm import tqdm

from hyperactive import Hyperactive

if sys.platform.startswith("win"):
    pytest.skip("skip these tests for windows", allow_module_level=True)


def objective_function(opt):
    """Return simple quadratic objective function for testing."""
    score = -opt["x1"] * opt["x1"]
    return score


search_space = {
    "x1": list(np.arange(-100, 101, 1)),
}


def test_n_jobs_0():
    """Test basic n_jobs functionality with 2 parallel jobs."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_jobs_1():
    """Test n_jobs functionality with 4 parallel jobs."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.run()

    assert len(hyper.results_list) == 4


def test_n_jobs_2():
    """Test n_jobs functionality with 8 parallel jobs."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=8)
    hyper.run()

    assert len(hyper.results_list) == 8


def test_n_jobs_3():
    """Test default n_jobs behavior (single job)."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15)
    hyper.run()


def test_n_jobs_5():
    """Test multiple searches with n_jobs=2 each."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)

    hyper.run()

    assert len(hyper.results_list) == 4


def test_n_jobs_6():
    """Test four searches with n_jobs=2 each."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)

    hyper.run()

    assert len(hyper.results_list) == 8


def test_n_jobs_7():
    """Test n_jobs=-1 (use all available cores)."""
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=-1)
    hyper.run()


def test_multiprocessing_0():
    """Test multiprocessing distribution backend."""
    hyper = Hyperactive(distribution="multiprocessing")
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()


def test_multiprocessing_1():
    """Test multiprocessing with custom initializer configuration."""
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
    """Test joblib distribution backend."""
    hyper = Hyperactive(distribution="joblib")
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()


def test_joblib_1():
    """Test custom joblib wrapper function."""
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
    """Test pathos distribution backend."""
    hyper = Hyperactive(distribution="pathos")
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()


def test_n_processes_0():
    """Test n_processes=1 with n_jobs=2."""
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_processes_1():
    """Test n_processes=2 with n_jobs=2."""
    hyper = Hyperactive(n_processes=2)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_processes_2():
    """Test n_processes=4 with n_jobs=2."""
    hyper = Hyperactive(n_processes=4)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=2)
    hyper.run()

    assert len(hyper.results_list) == 2


def test_n_processes_3():
    """Test n_processes=4 with n_jobs=3."""
    hyper = Hyperactive(n_processes=4)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=3)
    hyper.run()

    assert len(hyper.results_list) == 3


def test_n_processes_4():
    """Test n_processes=1 with n_jobs=4."""
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.run()

    assert len(hyper.results_list) == 4


def test_n_processes_5():
    """Test n_processes=1 with multiple searches having n_jobs=4."""
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.add_search(objective_function, search_space, n_iter=15, n_jobs=4)
    hyper.run()

    assert len(hyper.results_list) == 8
