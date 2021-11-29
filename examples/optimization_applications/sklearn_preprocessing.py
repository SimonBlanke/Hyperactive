import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(opt):
    model = GradientBoostingClassifier(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
    )

    X_pca = opt["decomposition"](X, opt)
    X_mod = np.hstack((X, X_pca))

    X_best = SelectKBest(f_classif, k=opt["k"]).fit_transform(X_mod, y)
    scores = cross_val_score(model, X_best, y, cv=3)

    return scores.mean()


def pca(X_, opt):
    X_ = PCA(n_components=opt["n_components"]).fit_transform(X_)

    return X_


def none(X_, opt):
    return X_


search_space = {
    "decomposition": [pca, none],
    "k": list(range(2, 30)),
    "n_components": list(range(1, 11)),
    "n_estimators": list(range(10, 100, 3)),
    "max_depth": list(range(2, 12)),
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=20)
hyper.run()
