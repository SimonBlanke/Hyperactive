import time
import numpy as np
from hyperactive import Hyperactive


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": list(np.arange(0, 100000, 1)),
}


def test_max_time_0():
    c_time1 = time.perf_counter()
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=1000000)
    hyper.run(max_time=0.1)
    diff_time1 = time.perf_counter() - c_time1

    assert diff_time1 < 1


def test_max_time_1():
    c_time1 = time.perf_counter()
    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=1000000)
    hyper.run(max_time=1)
    diff_time1 = time.perf_counter() - c_time1

    assert 0.3 < diff_time1 < 2
