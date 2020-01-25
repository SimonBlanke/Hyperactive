# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target

memory = False


def test_short_term_memory():
    def model1(para, X_train, y_train):
        model = DecisionTreeClassifier(criterion=para["criterion"])
        scores = cross_val_score(model, X_train, y_train, cv=5)

        return scores.mean()

    search_config = {model1: {"criterion": ["gini"]}}

    opt = Hyperactive(X, y, memory="short")
    opt.search(search_config, n_iter=1000)

    assert np.array(opt.eval_times[model1]).mean() < 1


def test_long_term_memory_with_data():
    def model2(para, X_train, y_train):
        model = DecisionTreeClassifier(
            criterion=para["criterion"], max_depth=para["max_depth"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {
        model2: {"criterion": ["gini", "entropy"], "max_depth": range(1, 11)}
    }

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)


def test_long_term_memory_without_data():
    def model3(para, X_train, y_train):
        model = DecisionTreeClassifier(
            criterion=para["criterion"], max_depth=para["max_depth"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {
        model3: {"criterion": ["gini", "entropy"], "max_depth": range(1, 11)}
    }

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config, n_iter=0)

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)


def test_long_term_memory_best_model():
    def model4(para, X_train, y_train):
        model = DecisionTreeClassifier(
            criterion=para["criterion"], max_depth=para["max_depth"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {
        model4: {"criterion": ["gini", "entropy"], "max_depth": range(1, 11)}
    }

    opt1 = Hyperactive(X, y, memory="long")
    opt1.search(search_config)

    best_para = opt1.results[model4]

    opt2 = Hyperactive(X, y, memory="long")
    opt2.search(search_config, n_iter=0, init_config=opt1.results)

    assert best_para == opt2.results[model4]


def test_long_term_memory_obj_storage():
    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

    def model(para, X_train, y_train):
        gpc = GaussianProcessClassifier(kernel=para["kernel"])
        scores = cross_val_score(gpc, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {model: {"kernel": [RBF(), Matern(), ConstantKernel()]}}

    opt1 = Hyperactive(X, y, memory="long")
    opt1.search(search_config)

    best_para = opt1.results[model]

    opt2 = Hyperactive(X, y, memory="long")
    opt2.search(search_config, n_iter=0, init_config=opt1.results)

    assert best_para == opt2.results[model]


def test_long_term_memory_search_space_expansion():
    def model5(para, X_train, y_train):
        model = DecisionTreeClassifier(criterion=para["criterion"])
        scores = cross_val_score(model, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {model5: {"criterion": ["gini", "entropy"]}}

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)

    def model5(para, X_train, y_train):
        model = DecisionTreeClassifier(
            criterion=para["criterion"], max_depth=para["max_depth"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {
        model5: {"criterion": ["gini", "entropy"], "max_depth": range(1, 11)}
    }

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)


def test_long_term_memory_search_space_reduction():
    def model6(para, X_train, y_train):
        model = DecisionTreeClassifier(
            criterion=para["criterion"], max_depth=para["max_depth"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {
        model6: {"criterion": ["gini", "entropy"], "max_depth": range(1, 11)}
    }

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)

    def model6(para, X_train, y_train):
        model = DecisionTreeClassifier(criterion=para["criterion"])
        scores = cross_val_score(model, X_train, y_train, cv=2)

        return scores.mean()

    search_config = {model6: {"criterion": ["gini", "entropy"]}}

    opt = Hyperactive(X, y, memory="long")
    opt.search(search_config)
