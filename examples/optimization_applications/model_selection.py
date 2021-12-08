from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.neural_network import MLPRegressor

from sklearn.datasets import load_diabetes
from hyperactive import Hyperactive

data = load_diabetes()
X, y = data.data, data.target


def model(opt):
    model_class = opt["regressor"]()
    model = model_class()
    scores = cross_val_score(model, X, y, cv=5)

    return scores.mean()


def SVR_f():
    return SVR


def KNeighborsRegressor_f():
    return KNeighborsRegressor


def GaussianProcessRegressor_f():
    return GaussianProcessRegressor


def DecisionTreeRegressor_f():
    return DecisionTreeRegressor


def GradientBoostingRegressor_f():
    return GradientBoostingRegressor


def RandomForestRegressor_f():
    return RandomForestRegressor


def ExtraTreesRegressor_f():
    return ExtraTreesRegressor


def MLPRegressor_f():
    return MLPRegressor


search_space = {
    "regressor": [
        SVR_f,
        KNeighborsRegressor_f,
        GaussianProcessRegressor_f,
        DecisionTreeRegressor_f,
        GradientBoostingRegressor_f,
        RandomForestRegressor_f,
        ExtraTreesRegressor_f,
        MLPRegressor_f,
    ],
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=50)
hyper.run()
