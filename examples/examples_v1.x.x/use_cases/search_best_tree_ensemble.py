# disables sklearn warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def GradientBoostingClassifier_(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


def RandomForestClassifier_(para, X, y):
    rfc = RandomForestClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(rfc, X, y, cv=3)

    return scores.mean()


def ExtraTreesClassifier_(para, X, y):
    etc = ExtraTreesClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(etc, X, y, cv=3)

    return scores.mean()


def XGBoost_(para, X, y):
    etc = XGBClassifier(n_estimators=para["n_estimators"], max_depth=para["max_depth"])
    scores = cross_val_score(etc, X, y, cv=3)

    return scores.mean()


search_config = {
    GradientBoostingClassifier_: {
        "n_estimators": range(50, 300, 5),
        "max_depth": range(2, 10),
        "min_samples_split": range(2, 20),
        "min_samples_leaf": range(2, 20),
    },
    RandomForestClassifier_: {
        "n_estimators": range(5, 100, 1),
        "max_depth": range(2, 20),
        "min_samples_split": range(2, 20),
        "min_samples_leaf": range(2, 20),
    },
    ExtraTreesClassifier_: {
        "n_estimators": range(50, 300, 5),
        "max_depth": range(2, 20),
        "min_samples_split": range(2, 20),
        "min_samples_leaf": range(2, 20),
    },
    XGBoost_: {"n_estimators": range(50, 300, 5), "max_depth": range(2, 20)},
}


opt = Hyperactive(search_config, n_jobs=4, n_iter=100)
opt.search(X, y)
