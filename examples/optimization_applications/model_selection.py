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

from sklearn.datasets import load_boston
from hyperactive import Hyperactive

data = load_boston()
X, y = data.data, data.target


def model(opt):
    gbr = opt["regressor"]()
    scores = cross_val_score(gbr, X, y, cv=5)

    return scores.mean()


search_space = {
    "regressor": [
        SVR,
        KNeighborsRegressor,
        GaussianProcessRegressor,
        DecisionTreeRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
        ExtraTreesRegressor,
        MLPRegressor,
    ],
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=50)
hyper.run()
