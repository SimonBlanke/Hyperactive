# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive
from hyperactive.memory import (
    delete_model,
    delete_model_dataset,
    connect_model_IDs,
    split_model_IDs,
    get_best_model,
    reset_memory,
)

data = load_iris()
X, y = data.data, data.target


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(criterion=para["criterion"])
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


def model1(para, X_train, y_train):
    model = DecisionTreeClassifier(max_depth=para["max_depth"])
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


def model2(para, X_train, y_train):
    model = DecisionTreeClassifier(max_depth=para["max_depth"])
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


search_config = {model: {"criterion": ["gini"]}}
search_config1 = {model1: {"max_depth": range(2, 500)}}
search_config2 = {model2: {"max_depth": range(2, 500)}}


def test_reset_memory():
    reset_memory(force_true=True)


def test_delete_model():
    delete_model(model)

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)

    delete_model(model)


def test_delete_model_dataset():
    delete_model_dataset(model, X, y)

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)

    delete_model_dataset(model, X, y)


def test_connect_model_IDs():
    delete_model(model1)
    delete_model(model2)

    connect_model_IDs(model1, model2)

    c_time = time.time()
    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config1, n_iter=1000)
    diff_time_0 = time.time() - c_time

    c_time = time.time()
    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config2, n_iter=1000)
    diff_time_1 = time.time() - c_time

    assert diff_time_0 / 2 > diff_time_1


def test_split_model_IDs():
    delete_model(model1)
    delete_model(model2)

    connect_model_IDs(model1, model2)

    split_model_IDs(model1, model2)

    c_time = time.time()
    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config1, n_iter=1000)
    diff_time_0 = time.time() - c_time

    c_time = time.time()
    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config2, n_iter=1000)
    diff_time_1 = time.time() - c_time

    assert diff_time_0 / 2 < diff_time_1


def test_get_best_model():
    score, search_config, init_config = get_best_model(X, y)
