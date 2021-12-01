"""
Hyperactive can perform optimizations of multiple different objective functions
in parallel. This can be done via multiprocessing, joblib or a custom wrapper-function.
The processes won't communicate with each other.

You can add as many searches to the optimization run (.add_search(...)) and 
run each of those searches n-times (n_jobs).

In the example below we are performing 4 searches in parallel:
    - model_etc one time
    - model_rfc one time
    - model_gbc two times

"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model_etc(opt):
    etc = ExtraTreesClassifier(
        n_estimators=opt["n_estimators"],
        criterion=opt["criterion"],
        max_features=opt["max_features"],
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
        bootstrap=opt["bootstrap"],
    )
    scores = cross_val_score(etc, X, y, cv=3)

    return scores.mean()


def model_rfc(opt):
    rfc = RandomForestClassifier(
        n_estimators=opt["n_estimators"],
        criterion=opt["criterion"],
        max_features=opt["max_features"],
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
        bootstrap=opt["bootstrap"],
    )
    scores = cross_val_score(rfc, X, y, cv=3)

    return scores.mean()


def model_gbc(opt):
    gbc = GradientBoostingClassifier(
        n_estimators=opt["n_estimators"],
        learning_rate=opt["learning_rate"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
        subsample=opt["subsample"],
        max_features=opt["max_features"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_space_etc = {
    "n_estimators": list(range(10, 200, 10)),
    "criterion": ["gini", "entropy"],
    "max_features": list(np.arange(0.05, 1.01, 0.05)),
    "min_samples_split": list(range(2, 21)),
    "min_samples_leaf": list(range(1, 21)),
    "bootstrap": [True, False],
}


search_space_rfc = {
    "n_estimators": list(range(10, 200, 10)),
    "criterion": ["gini", "entropy"],
    "max_features": list(np.arange(0.05, 1.01, 0.05)),
    "min_samples_split": list(range(2, 21)),
    "min_samples_leaf": list(range(1, 21)),
    "bootstrap": [True, False],
}


search_space_gbc = {
    "n_estimators": list(range(10, 200, 10)),
    "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
    "max_depth": list(range(1, 11)),
    "min_samples_split": list(range(2, 21)),
    "min_samples_leaf": list(range(1, 21)),
    "subsample": list(np.arange(0.05, 1.01, 0.05)),
    "max_features": list(np.arange(0.05, 1.01, 0.05)),
}


hyper = Hyperactive()
hyper.add_search(model_etc, search_space_etc, n_iter=50)
hyper.add_search(model_rfc, search_space_rfc, n_iter=50)
hyper.add_search(model_gbc, search_space_gbc, n_iter=50, n_jobs=2)
hyper.run(max_time=5)

search_data_etc = hyper.search_data(model_etc)
search_data_rfc = hyper.search_data(model_rfc)
search_data_gbc = hyper.search_data(model_gbc)

print("\n ExtraTreesClassifier search data \n", search_data_etc)
print("\n GradientBoostingClassifier search data \n", search_data_gbc)
