# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from hyperactive import Hyperactive

data = load_iris()
X, y = data.data, data.target


def model(para, X, y):
    dtc = DecisionTreeClassifier(
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(dtc, X, y, cv=2)

    return scores.mean()


search_space = {
    "max_depth": range(1, 21),
    "min_samples_split": range(2, 21),
    "min_samples_leaf": range(1, 21),
}


def _base_test(search, opt_args={}, time=None):
    hyper = Hyperactive(X, y, **opt_args)
    hyper.add_search(**search)
    hyper.run(time)


def test_model_key():
    search = {
        "model_": model,
        "search_space": search_space,
    }
    with pytest.raises(TypeError):
        _base_test(search)


def test_model_value():
    search = {
        "model": 1,
        "search_space": search_space,
    }
    with pytest.raises(ValueError):
        _base_test(search)


def test_search_space_key():
    search = {
        "model": model,
        "search_space_": search_space,
    }
    with pytest.raises(TypeError):
        _base_test(search)


def test_search_space_value():
    search = {
        "model": model,
        "search_space": 1,
    }
    with pytest.raises(ValueError):
        _base_test(search)


def test_memory_key():
    search = {
        "model": model,
        "search_space": search_space,
        "memory_": "short",
    }
    with pytest.raises(TypeError):
        _base_test(search)


def test_memory_value():
    search = {
        "model": model,
        "search_space": search_space,
        "memory": 1,
    }
    with pytest.raises(ValueError):
        _base_test(search)


def test_optimizer_key():
    search = {
        "model": model,
        "search_space": search_space,
        "optimizer_": 1,
    }
    with pytest.raises(TypeError):
        _base_test(search)


def test_optimizer_value():
    search = {
        "model": model,
        "search_space": search_space,
        "optimizer": 1,
    }
    with pytest.raises(ValueError):
        _base_test(search)


def test_n_iter_key():
    search = {
        "model": model,
        "search_space": search_space,
        "n_iter_": 10,
    }
    with pytest.raises(TypeError):
        _base_test(search)


def test_n_iter_value():
    search = {
        "model": model,
        "search_space": search_space,
        "n_iter": 0.1,
    }
    with pytest.raises(ValueError):
        _base_test(search)


def test_n_jobs_key():
    search = {
        "model": model,
        "search_space": search_space,
        "n_jobs_": 1,
    }
    with pytest.raises(TypeError):
        _base_test(search)


def test_n_jobs_value():
    search = {
        "model": model,
        "search_space": search_space,
        "n_jobs": 0.1,
    }
    with pytest.raises(ValueError):
        _base_test(search)


def test_init_para_key():
    search = {
        "model": model,
        "search_space": search_space,
        "init_para_": {},
    }
    with pytest.raises(TypeError):
        _base_test(search)


def test_init_para_value():
    search = {
        "model": model,
        "search_space": search_space,
        "init_para": 1,
    }
    with pytest.raises(ValueError):
        _base_test(search)

