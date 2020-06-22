# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"], max_depth=para["max_depth"]
    )
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


search_config = {model: {"criterion": ["gini", "entropy"], "max_depth": range(1, 21)}}


def test_checks_X():
    with pytest.raises(ValueError):
        opt = Hyperactive(1, y)


def test_checks_y():
    with pytest.raises(ValueError):
        opt = Hyperactive(X, 1)


def test_checks_memory():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, memory=None)


def test_checks_random_state():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, random_state=None)


def test_checks_verbosity():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, verbosity=None)


def test_checks_search_config():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, verbosity=None)
        opt.search(1)

    search_config = {1}

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, verbosity=None)
        opt.search(search_config)

    search_config = {model: 1}

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, verbosity=None)
        opt.search(search_config)


def test_checks_n_iter():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y)
        opt.search(search_config, n_iter=0.1)


def test_checks_max_time():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y)
        opt.search(search_config, max_time="1")


def test_checks_optimizer():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y)
        opt.search(search_config, optimizer=1)


def test_checks_n_jobs():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y)
        opt.search(search_config, n_jobs=0.1)


def test_checks_scheduler():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y)
        opt.search(search_config, scheduler=1)


def test_checks_init_config():

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, verbosity=None)
        opt.search(search_config, init_config=1)

    init_config = {1}

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, verbosity=None)
        opt.search(search_config, init_config=init_config)

    init_config = {model: 1}

    with pytest.raises(ValueError):
        opt = Hyperactive(X, y, verbosity=None)
        opt.search(search_config, init_config=init_config)
