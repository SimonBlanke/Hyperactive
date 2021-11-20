import time
import pytest
import numpy as np
import pandas as pd

from hyperactive import Hyperactive


def objective_function_0(opt):
    score = -opt["x1"] * opt["x1"]
    return score


search_space_0 = {
    "x1": list(np.arange(-5, 6, 1)),
}
search_space_1 = {
    "x1": list(np.arange(0, 6, 1)),
}
search_space_2 = {
    "x1": list(np.arange(-5, 1, 1)),
}


search_space_3 = {
    "x1": list(np.arange(-1, 1, 0.1)),
}
search_space_4 = {
    "x1": list(np.arange(-1, 0, 0.1)),
}
search_space_5 = {
    "x1": list(np.arange(0, 1, 0.1)),
}


search_space_para_0 = [
    (search_space_0),
    (search_space_1),
    (search_space_2),
    (search_space_3),
    (search_space_4),
    (search_space_5),
]


@pytest.mark.parametrize("search_space", search_space_para_0)
def test_trafo_0(search_space):
    hyper = Hyperactive()
    hyper.add_search(objective_function_0, search_space, n_iter=25)
    hyper.run()

    for value in hyper.results(objective_function_0)["x1"].values:
        if value not in search_space["x1"]:
            assert False


# ----------------- # Test if memory warm starts do work as intended


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X, y = data.data, data.target


def objective_function_1(opt):
    dtc = DecisionTreeClassifier(min_samples_split=opt["min_samples_split"])
    scores = cross_val_score(dtc, X, y, cv=10)
    time.sleep(0.1)

    return scores.mean()


search_space_0 = {
    "min_samples_split": list(np.arange(2, 12)),
}

search_space_1 = {
    "min_samples_split": list(np.arange(12, 22)),
}

search_space_2 = {
    "min_samples_split": list(np.arange(22, 32)),
}

memory_dict = {"min_samples_split": range(2, 12), "score": range(2, 12)}
memory_warm_start_0 = pd.DataFrame(memory_dict)

memory_dict = {"min_samples_split": range(12, 22), "score": range(12, 22)}
memory_warm_start_1 = pd.DataFrame(memory_dict)

memory_dict = {"min_samples_split": range(22, 32), "score": range(22, 32)}
memory_warm_start_2 = pd.DataFrame(memory_dict)

search_space_para_1 = [
    (search_space_0, memory_warm_start_0),
    (search_space_1, memory_warm_start_1),
    (search_space_2, memory_warm_start_2),
]

random_state_para_0 = [
    (0),
    (1),
    (2),
    (3),
    (4),
]


@pytest.mark.parametrize("random_state", random_state_para_0)
@pytest.mark.parametrize("search_space, memory_warm_start", search_space_para_1)
def test_trafo_1(random_state, search_space, memory_warm_start):
    search_space = search_space
    memory_warm_start = memory_warm_start

    c_time_0 = time.perf_counter()
    hyper = Hyperactive()
    hyper.add_search(
        objective_function_1,
        search_space,
        n_iter=10,
        random_state=random_state,
        initialize={"random": 1},
    )
    hyper.run()
    d_time_0 = time.perf_counter() - c_time_0

    c_time_1 = time.perf_counter()
    hyper = Hyperactive()
    hyper.add_search(
        objective_function_1,
        search_space,
        n_iter=10,
        random_state=random_state,
        initialize={"random": 1},
        memory_warm_start=memory_warm_start,
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time_1

    assert d_time_1 < d_time_0 * 0.5


# ----------------- # Test if wrong memory warm starts do not work as intended
""" test is possible in future gfo versions
search_space_0 = {
    "min_samples_split": list(np.arange(2, 12)),
}

search_space_1 = {
    "min_samples_split": list(np.arange(12, 22)),
}

search_space_2 = {
    "min_samples_split": list(np.arange(22, 32)),
}

memory_dict = {"min_samples_split": range(12, 22), "score": range(2, 12)}
memory_warm_start_0 = pd.DataFrame(memory_dict)

memory_dict = {"min_samples_split": range(22, 32), "score": range(12, 22)}
memory_warm_start_1 = pd.DataFrame(memory_dict)

memory_dict = {"min_samples_split": range(2, 12), "score": range(22, 32)}
memory_warm_start_2 = pd.DataFrame(memory_dict)

search_space_para_2 = [
    (search_space_0, memory_warm_start_0),
    (search_space_1, memory_warm_start_1),
    (search_space_2, memory_warm_start_2),
]

random_state_para_0 = [
    (0),
    (1),
    (2),
    (3),
    (4),
]


@pytest.mark.parametrize("random_state", random_state_para_0)
@pytest.mark.parametrize("search_space, memory_warm_start", search_space_para_2)
def test_trafo_2(random_state, search_space, memory_warm_start):
    search_space = search_space
    memory_warm_start = memory_warm_start

    c_time_0 = time.perf_counter()
    hyper = Hyperactive()
    hyper.add_search(
        objective_function_1,
        search_space,
        n_iter=25,
        random_state=random_state,
        initialize={"random": 1},
    )
    hyper.run()
    d_time_0 = time.perf_counter() - c_time_0

    c_time_1 = time.perf_counter()
    hyper = Hyperactive()
    hyper.add_search(
        objective_function_1,
        search_space,
        n_iter=25,
        random_state=random_state,
        initialize={"random": 1},
        memory_warm_start=memory_warm_start,
    )
    hyper.run()
    d_time_1 = time.perf_counter() - c_time_1

    assert not (d_time_1 < d_time_0 * 0.8)
"""
