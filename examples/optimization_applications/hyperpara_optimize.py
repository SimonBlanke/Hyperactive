"""
This example shows the original purpose of Hyperactive.
You can search for any number of hyperparameters and Hyperactive
will return the best one after the optimization run.

"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from hyperactive import Hyperactive

data = load_wine()
X, y = data.data, data.target


def model(opt):
    gbr = GradientBoostingClassifier(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
        criterion=opt["criterion"],
    )
    scores = cross_val_score(gbr, X, y, cv=4)

    return scores.mean()


search_space = {
    "n_estimators": list(range(10, 150, 5)),
    "max_depth": list(range(2, 12)),
    "min_samples_split": list(range(2, 25)),
    "min_samples_leaf": list(range(1, 25)),
    "criterion": ["friedman_mse", "mse", "mae"],
    "subsample": list(np.arange(0.1, 3, 0.1)),
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=40)
hyper.run()
