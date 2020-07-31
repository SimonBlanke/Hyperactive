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


def model_etc(para, X, y):
    etc = ExtraTreesClassifier(
        n_estimators=para["n_estimators"],
        criterion=para["criterion"],
        max_features=para["max_features"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
        bootstrap=para["bootstrap"],
    )
    scores = cross_val_score(etc, X, y, cv=3)

    return scores.mean()


def model_rfc(para, X, y):
    rfc = RandomForestClassifier(
        n_estimators=para["n_estimators"],
        criterion=para["criterion"],
        max_features=para["max_features"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
        bootstrap=para["bootstrap"],
    )
    scores = cross_val_score(rfc, X, y, cv=3)

    return scores.mean()


def model_gbc(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        learning_rate=para["learning_rate"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
        subsample=para["subsample"],
        max_features=para["max_features"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


def model_xgb(para, X, y):
    xgb = XGBClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        learning_rate=para["learning_rate"],
    )
    scores = cross_val_score(xgb, X, y, cv=3)

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

search_space_xgb = {
    "n_estimators": list(range(10, 200, 10)),
    "max_depth": list(range(2, 12)),
    "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
}

hyper = Hyperactive(X, y)
hyper.add_search(model_etc, search_space_etc, n_iter=50)
hyper.add_search(model_rfc, search_space_rfc, n_iter=50)
hyper.add_search(model_gbc, search_space_gbc, n_iter=50)
hyper.add_search(model_xgb, search_space_xgb, n_iter=50)
hyper.run(max_time=10)
