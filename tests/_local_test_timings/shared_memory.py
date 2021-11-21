import time
import numpy as np
import pandas as pd

from hyperactive import Hyperactive


def model(opt):
    time.sleep(0.001)
    return 0


def model1(opt):
    time.sleep(0.001)
    return 0


def model2(opt):
    time.sleep(0.001)
    return 0


def model3(opt):
    time.sleep(0.001)
    return 0


def model4(opt):
    time.sleep(0.001)
    return 0


search_space = {
    "x1": list(range(2, 200)),
}


def test_shared_memory_0():
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory="share",
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time

    n_jobs = 4
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=n_jobs,
        memory="share",
    )
    hyper.run()
    d_time_2 = time.perf_counter() - c_time
    d_time_2 = d_time_2 / n_jobs

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time = d_time_1 / d_time_2

    assert d_time > 1.4


def test_shared_memory_1():
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory=True,
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time

    n_jobs = 4
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=n_jobs,
        memory=True,
    )
    hyper.run()
    d_time_2 = time.perf_counter() - c_time
    d_time_2 = d_time_2 / n_jobs

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time = d_time_1 / d_time_2

    assert d_time > 0.85
    assert d_time < 1.15


def test_shared_memory_2():
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time

    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory="share",
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory="share",
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory="share",
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory="share",
    )
    hyper.run()
    d_time_2 = time.perf_counter() - c_time
    d_time_2 = d_time_2 / 4

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time = d_time_1 / d_time_2

    assert d_time > 1.2


def test_shared_memory_3():
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time

    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory=True,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory=True,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory=True,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
        memory=True,
    )
    hyper.run()
    d_time_2 = time.perf_counter() - c_time
    d_time_2 = d_time_2 / 4

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time = d_time_1 / d_time_2

    assert d_time > 0.85
    assert d_time < 1.15


def test_shared_memory_warm_start_0():
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time

    search_data0 = hyper.results(model)

    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=4,
        memory_warm_start=search_data0,
        memory="share",
    )
    hyper.run()
    d_time_2 = time.perf_counter() - c_time
    d_time_2 = d_time_2 / 4

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time = d_time_1 / d_time_2

    assert d_time > 1.4


def test_shared_memory_warm_start_1():
    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time

    search_data0 = hyper.results(model)

    c_time = time.perf_counter()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=300,
        n_jobs=4,
        memory_warm_start=search_data0,
        memory=True,
    )
    hyper.run()
    d_time_2 = time.perf_counter() - c_time
    d_time_2 = d_time_2 / 4

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time = d_time_1 / d_time_2

    assert d_time > 2.3
