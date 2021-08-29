import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor


from hyperactive import Hyperactive


data = load_boston()
X, y = data.data, data.target


cv = 10


def model(opt):
    gbr = DecisionTreeRegressor(
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=cv)

    return scores.mean()


def model1(opt):
    gbr = DecisionTreeRegressor(
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=cv)

    return scores.mean()


def model2(opt):
    gbr = DecisionTreeRegressor(
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=cv)

    return scores.mean()


def model3(opt):
    gbr = DecisionTreeRegressor(
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=cv)

    return scores.mean()


def model4(opt):
    gbr = DecisionTreeRegressor(
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=cv)

    return scores.mean()


search_space = {
    "min_samples_split": list(range(2, 100)),
}


def test_shared_memory_0():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory=True,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=4,
        memory=True,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_2 - d_time_1 < 0


def test_shared_memory_1():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=4,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_1 / d_time_2 > 1.1


def test_shared_memory_2():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_1 / d_time_2 > 1.1


def test_shared_memory_3():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model1,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model2,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model1,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model2,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model3,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model4,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_1 / d_time_2 < 1.2


def test_shared_memory_warm_start_0():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    search_data0 = hyper.results(model)

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=4,
        memory_warm_start=search_data0,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_2 * 1.4 - d_time_1 < 0


def test_shared_memory_warm_start_1():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    search_data0 = hyper.results(model)

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=4,
        memory_warm_start=search_data0,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_1 / d_time_2 > 1.1


def test_shared_memory_warm_start_2():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    search_data0 = hyper.results(model)

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_1 / d_time_2 > 1.1


def test_shared_memory_warm_start_3():
    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model1,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model2,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model3,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.add_search(
        model4,
        search_space,
        n_iter=100,
        n_jobs=1,
    )
    hyper.run()
    d_time_1 = time.time() - c_time

    search_data0 = hyper.results(model1)

    c_time = time.time()
    hyper = Hyperactive(n_processes=1)
    hyper.add_search(
        model,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.add_search(
        model1,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.add_search(
        model2,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.add_search(
        model3,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.add_search(
        model4,
        search_space,
        n_iter=100,
        n_jobs=1,
        memory_warm_start=search_data0,
    )
    hyper.run()
    d_time_2 = time.time() - c_time

    print("\n d_time_1 \n", d_time_1)
    print("\n d_time_2 \n", d_time_2)

    d_time_2 = d_time_2 / 2

    assert d_time_1 / d_time_2 > 1.4
